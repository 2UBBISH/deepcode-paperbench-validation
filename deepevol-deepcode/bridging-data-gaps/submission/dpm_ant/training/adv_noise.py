"""Adversarial Noise (AN) selection for DPMs-ANT -- inner maximization (Eq. 7).

Paper
-----
Section 4.2 "Adversarial Noise Selection" and Equation (7):

.. math::

    \\epsilon^{j+1} = \\operatorname{Norm}\\Big(\\epsilon^{j}
        + \\omega \\nabla_{\\epsilon^{j}}
          \\big\\|\\epsilon^{j}
          - \\epsilon_{\\theta}\\big(\\sqrt{\\bar{\\alpha}_t}x_0
            + \\sqrt{1-\\bar{\\alpha}_t}\\,\\epsilon^{j}, t\\big)\\big\\|^2\\Big),

where ``j in {0, ..., J-1}``, ``omega`` is the "learning rate" of the negative
loss (gradient *ascent*), ``Norm(.)`` approximately ensures that the mean and
standard deviation of ``epsilon^{j+1}`` are ``0`` and ``I`` respectively, and
``epsilon^0 ~ N(0, I)``.

As stated in the paper: "the similarity-guided term is disregarded, as this term
is hard to compute differential and is almost unchanged in the process" -- so the
inner loop below contains *only* the denoising residual, never the classifier
guidance term.

The outer adaptation step (Algorithm 1 / Eq. 8) then uses ``epsilon* = epsilon^J``
and ``x_t* = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon*`` to update
only the adaptor parameters ``psi`` (see ``ant_trainer.py``).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from ..diffusion.schedule import NoiseSchedule, build_schedule

try:  # pragma: no cover - convenience import; a local fallback is provided
    from ..diffusion.gaussian_diffusion import extract as _gd_extract
except Exception:  # pragma: no cover
    _gd_extract = None


__all__ = [
    "AdversarialNoiseSelector",
    "select_adversarial_noise",
    "normalize_noise",
    "build_adv_noise_selector",
    "build_xt_from_noise",
]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _extract(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Gather per-sample schedule values at indices ``t`` and broadcast to ``ndim`` dims."""
    if not torch.is_tensor(arr):
        arr = torch.as_tensor(arr, dtype=torch.float32)
    t = t.to(device=arr.device)
    if t.dim() == 0:
        t = t.reshape(1)
    out = arr[t.reshape(-1).long()]
    while out.dim() < ndim:
        out = out.unsqueeze(-1)
    return out


def _resolve_callable(model: Any) -> Callable[..., torch.Tensor]:
    """Return a callable ``f(x_t, t) -> eps_theta`` for any supported model object.

    Accepts either a plain callable (``nn.Module.__call__``), an object exposing
    ``epsilon_theta`` (``FrozenDDPMUNet`` / ``FrozenLDMUNet``), or an object
    exposing ``predict_noise`` (some wrappers).
    """
    if model is None:
        raise ValueError("An adversarial-noise selector needs a noise-prediction model.")
    for attr in ("epsilon_theta", "predict_noise", "predict_eps"):
        fn = getattr(model, attr, None)
        if callable(fn):
            return fn
    if callable(model):
        return model
    raise TypeError(f"Model of type {type(model)} is not callable for noise prediction.")


def _unwrap_prediction(pred: Any) -> torch.Tensor:
    """Take the noise component if the model returns ``(eps, sigma)`` or a dict/tuple."""
    if isinstance(pred, (tuple, list)):
        return pred[0]
    if isinstance(pred, dict):  # pragma: no cover - defensive
        for key in ("eps", "epsilon", "pred", "out"):
            if key in pred:
                return pred[key]
    return pred


def _set_requires_grad(module: Any, flag: bool) -> Optional[Dict[str, bool]]:
    """Set ``requires_grad`` for all parameters of ``module``; return the previous state."""
    params = getattr(module, "parameters", None)
    if not callable(params):
        return None
    prev: Dict[str, bool] = {}
    for name, p in module.named_parameters() if hasattr(module, "named_parameters") else []:
        prev[name] = bool(p.requires_grad)
        p.requires_grad_(flag)
    return prev


def _restore_requires_grad(module: Any, prev: Optional[Dict[str, bool]]) -> None:
    if not prev:
        return
    named = dict(module.named_parameters()) if hasattr(module, "named_parameters") else {}
    for name, flag in prev.items():
        if name in named:
            named[name].requires_grad_(flag)


# --------------------------------------------------------------------------------------
# Norm(.)
# --------------------------------------------------------------------------------------
def normalize_noise(
    noise: torch.Tensor,
    mode: str = "per_sample",
    eps: float = 1e-5,
    unbiased: bool = False,
) -> torch.Tensor:
    """``Norm(.)`` from Eq. (7): re-standardize noise to approximately mean 0 / std I.

    Modes
    -----
    ``per_sample``
        Standardize *each sample independently* over all remaining elements
        (the paper's default interpretation and the setting used in the paper's
        experiments): ``(e - e.mean()) / (e.std() + eps)``.
    ``per_channel``
        Standardize each sample and channel independently (over spatial elements).
    ``per_batch``
        Standardize over the whole batch (single global mean/std).
    ``rms``
        L2-style normalization ``e / RMS(e)`` which also matches mean 0 only in
        expectation; exposed as a fallback for sensitivity checks.

    Notes
    -----
    The statistics are computed *without* tracking gradients (the noise is a
    fresh leaf/derived tensor for the next ascent iteration anyway).
    """
    if mode is None:
        mode = "per_sample"
    mode = str(mode).lower()
    with torch.no_grad():
        if mode in ("per_sample", "sample", "global_per_sample"):
            dims = tuple(range(1, noise.dim())) or None
        elif mode in ("per_channel", "channel"):
            dims = tuple(range(2, noise.dim())) if noise.dim() > 2 else None
        elif mode in ("per_batch", "batch", "all"):
            dims = None
        elif mode in ("rms", "l2", "unit_rms"):
            rms = noise.reshape(noise.shape[0], -1).pow(2).mean(dim=1).sqrt()
            view = rms.reshape(-1, *([1] * (noise.dim() - 1)))
            return noise / (view + eps)
        else:
            raise ValueError(
                f"Unknown Norm(.) mode {mode!r}; use 'per_sample', 'per_channel', 'per_batch' or 'rms'."
            )
        if dims is None:
            mean = noise.mean()
            std = noise.std(unbiased=unbiased)
        else:
            mean = noise.mean(dim=dims, keepdim=True)
            std = noise.std(dim=dims, unbiased=unbiased, keepdim=True)
    return (noise - mean) / (std + eps)


def build_xt_from_noise(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """Reparameterized forward process ``x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps``."""
    ndim = x0.dim()
    sqrt_ab = torch.sqrt(schedule.ab(t, ndim))
    sqrt_1m_ab = torch.sqrt(1.0 - schedule.ab(t, ndim))
    return sqrt_ab * x0 + sqrt_1m_ab * noise


# --------------------------------------------------------------------------------------
# selector
# --------------------------------------------------------------------------------------
class AdversarialNoiseSelector:
    """Finite-step gradient ascent solver for the inner ``max_epsilon`` of Eq. (7).

    Parameters
    ----------
    model:
        Frozen pretrained noise-prediction network (``epsilon_theta``). Anything
        exposing ``epsilon_theta(x_t, t)`` or callable as ``model(x_t, t)`` works.
    schedule:
        A :class:`~dpm_ant.diffusion.schedule.NoiseSchedule`. Built from
        ``num_timesteps`` / ``schedule`` / ``**schedule_kwargs`` when omitted.
    J:
        Number of inner ascent steps (paper default ``10``).
    omega:
        Inner "learning rate" of the negative loss (paper default ``0.02``).
    norm:
        ``Norm(.)`` mode, one of ``per_sample`` (default), ``per_channel``,
        ``per_batch``, ``rms``.
    loss_reduction:
        ``"sum"`` (paper: the squared **norm** ``||.||^2``, summed over all
        elements and averaged over the batch) or ``"mean"``.
    grad_normalize:
        If ``True``, use the unit-norm gradient direction before scaling by
        ``omega`` (helps when comparing losses across backbones); default
        ``False`` to stay faithful to Eq. (7).
    normalize_each_step:
        If ``False``, ``Norm(.)`` is applied once at the end instead of after
        every ascent step (ablation switch).
    clip_grad / clip_noise:
        Optional safeguards for numerical stability.
    """

    def __init__(
        self,
        model: Any = None,
        schedule: Optional[NoiseSchedule] = None,
        num_timesteps: int = 1000,
        schedule_name: str = "linear",
        J: int = 10,
        omega: float = 0.02,
        norm: str = "per_sample",
        loss_reduction: str = "sum",
        grad_normalize: bool = False,
        normalize_each_step: bool = True,
        clip_grad: Optional[float] = None,
        clip_noise: Optional[float] = None,
        detach_model_params: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        **schedule_kwargs: Any,
    ) -> None:
        self.model = model
        self.schedule = schedule if schedule is not None else build_schedule(
            {
                "num_timesteps": num_timesteps,
                "schedule": schedule_name,
                **schedule_kwargs,
            }
        )
        self.num_timesteps = int(self.schedule.num_timesteps)
        self.device = device if device is not None else getattr(self.schedule, "device", None)
        self.dtype = dtype

        self.J = int(J)
        self.omega = float(omega)
        self.norm = norm
        self.loss_reduction = loss_reduction
        self.grad_normalize = bool(grad_normalize)
        self.normalize_each_step = bool(normalize_each_step)
        self.clip_grad = clip_grad
        self.clip_noise = clip_noise
        self.detach_model_params = bool(detach_model_params)
        self.history: list = []

    # -- property-style accessors used by the trainer / toy experiment -------------
    def sqrt_alpha_bar(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return torch.sqrt(self.schedule.ab(t, ndim))

    def sqrt_one_minus_alpha_bar(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return torch.sqrt(1.0 - self.schedule.ab(t, ndim))

    def build_x_t(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return build_xt_from_noise(x0, t, noise, self.schedule)

    def sample_timesteps(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """``t ~ Uniform({1, ..., T})`` (Algorithm 1)."""
        return torch.randint(
            1,
            self.num_timesteps + 1,
            (batch_size,),
            device=device if device is not None else self.device,
            generator=generator,
        )

    # -- the inner objective -------------------------------------------------------
    def inner_loss(
        self,
        noise: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        model: Any = None,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        """``|| eps^j - eps_theta(x_t^j, t) ||^2`` averaged over the batch (Eq. 7)."""
        fn = _resolve_callable(model if model is not None else self.model)
        reduction = reduction or self.loss_reduction
        x_t = self.build_x_t(x0, t, noise)
        pred = _unwrap_prediction(fn(x_t, t))
        residual = (noise - pred).float()
        flat = residual.reshape(residual.shape[0], -1)
        if reduction == "mean":
            return flat.pow(2).mean()
        return flat.pow(2).sum(dim=1).mean()

    def ascent_step(
        self,
        noise: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        model: Any = None,
        omega: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One inner gradient-ascent step of Eq. (7).

        Returns ``(eps_next, loss)`` where ``loss`` is the detached inner loss used
        to build the ascent direction (useful to verify that the inner ascent
        increases the objective).
        """
        omega = self.omega if omega is None else float(omega)
        fn = _resolve_callable(model if model is not None else self.model)
        with torch.enable_grad():
            noise = noise.detach().requires_grad_(True)
            loss = self.inner_loss(noise, x0, t, model=fn)
            grad = torch.autograd.grad(loss, noise, retain_graph=False, create_graph=False)[0]
        if self.grad_normalize:
            gflat = grad.reshape(grad.shape[0], -1)
            grad = (grad / (gflat.norm(dim=1).reshape(-1, *([1] * (grad.dim() - 1))) + 1e-8))
        if self.clip_grad is not None:
            grad = grad.clamp(-float(self.clip_grad), float(self.clip_grad))
        with torch.no_grad():
            noise_next = noise.detach() + omega * grad
            if self.normalize_each_step:
                noise_next = normalize_noise(noise_next, mode=self.norm)
                if self.clip_noise is not None:
                    noise_next = noise_next.clamp(-float(self.clip_noise), float(self.clip_noise))
        return noise_next.detach(), loss.detach()

    # -- full inner loop -----------------------------------------------------------
    def select(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model: Any = None,
        J: Optional[int] = None,
        omega: Optional[float] = None,
        norm: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
        return_xt: bool = True,
        return_info: bool = False,
        record_history: bool = True,
    ):
        """Run ``J`` inner ascent steps and return the worst-case noise.

        Parameters
        ----------
        x0:
            Source-domain training batch (target fine-tuning set actually drives the
            transfer; Algorithm 1 samples ``x_0 ~ q(x_0)`` from the target 10-shot set).
        t:
            Optional timesteps in ``{1, ..., T}``. Sampled uniformly when ``None``.
        model:
            Override for the model passed at construction.
        J, omega, norm:
            Optional overrides of the construction-time defaults.

        Returns
        -------
        ``(eps_star, x_t_star)`` by default, or ``(eps_star, x_t_star, info)`` with
        ``info = {"losses": [...], "eps": eps_star, "t": t}`` when ``return_info=True``.
        """
        model = model if model is not None else self.model
        fn = _resolve_callable(model)
        J = self.J if J is None else int(J)
        omega = self.omega if omega is None else float(omega)
        norm = norm or self.norm

        if t is None:
            t = self.sample_timesteps(x0.shape[0], device=x0.device, generator=generator)
        else:
            t = torch.as_tensor(t, device=x0.device)
            if t.dim() == 0:
                t = t.reshape(1).expand(x0.shape[0])
            t = t.reshape(-1).long()

        if init_noise is None:
            init_noise = torch.randn(
                x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
            )
        noise = init_noise.detach().clone()

        prev_state = None
        if self.detach_model_params and self.clip_grad is not None:
            # (model params never receive gradients from ``autograd.grad(loss, noise)``;
            #  freezing is only enforced for very large graphs)
            prev_state = _set_requires_grad(model, False)
        try:
            losses = []
            for _ in range(max(J, 0)):
                noise, loss = self.ascent_step(noise, x0, t, model=fn, omega=omega)
                losses.append(float(loss))
        finally:
            if prev_state is not None:
                _restore_requires_grad(model, prev_state)

        eps_star = noise.detach()
        if not self.normalize_each_step:
            eps_star = normalize_noise(eps_star, mode=norm)
        x_t_star = self.build_x_t(x0, t, eps_star).detach()
        if record_history:
            self.history.append(losses)

        if not return_xt:
            out = (eps_star,)
        else:
            out = (eps_star, x_t_star)
        if return_info:
            out = out + ({"losses": losses, "t": t, "omega": omega, "J": J, "norm": norm},)
        return out

    __call__ = select

    # convenience for the toy experiment: expose the raw loss for a fixed noise
    def loss_of(self, noise: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, model: Any = None):
        with torch.no_grad():
            return float(self.inner_loss(noise, x0, t, model=model))


# --------------------------------------------------------------------------------------
# functional wrappers / factories
# --------------------------------------------------------------------------------------
def select_adversarial_noise(
    model: Any,
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: Optional[NoiseSchedule] = None,
    J: int = 10,
    omega: float = 0.02,
    norm: str = "per_sample",
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-shot functional API: return ``(eps*, x_t*)`` from Eq. (7) / Algorithm 1."""
    selector = AdversarialNoiseSelector(
        model=model, schedule=schedule, J=J, omega=omega, norm=norm, **kwargs
    )
    eps_star, x_t_star = selector.select(x0, t=t)
    return eps_star, x_t_star


def build_adv_noise_selector(
    cfg: Optional[Dict[str, Any]] = None,
    model: Any = None,
    backbone: str = "ddpm",
    task: Optional[str] = None,
    **overrides: Any,
) -> AdversarialNoiseSelector:
    """Build a selector from a (possibly nested) config dict.

    Recognized keys (searched in ``cfg['ant']`` first, then the flat dict; a
    ``defaults`` block and a ``tasks.<name>`` block are also honored so that
    ``configs/default.yaml`` / ``configs/per_task.yaml`` can be passed verbatim):

    ``J``, ``omega``, ``norm``, ``loss_reduction``, ``grad_normalize``,
    ``normalize_each_step``, ``clip_grad``, ``clip_noise``, ``detach_model_params``,
    plus ``diffusion.num_timesteps`` / ``diffusion.schedule`` / ``diffusion.eta``.
    """
    cfg = dict(cfg or {})
    merged: Dict[str, Any] = {}

    def _absorb(d: Any) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    continue
                merged.setdefault(k, v)

    _absorb(cfg)
    for block in ("defaults", "ant", "diffusion", "sampling"):
        _absorb(cfg.get(block))
    if task is not None and isinstance(cfg.get("tasks"), dict):
        _absorb(cfg["tasks"].get(task))
    merged.update({k: v for k, v in overrides.items() if v is not None})

    backbone = str(merged.pop("backbone", backbone)).lower()
    num_timesteps = int(merged.pop("num_timesteps", merged.pop("T", 1000)))
    schedule_name = merged.pop("schedule", merged.pop("beta_schedule", "linear"))
    eta = float(merged.pop("eta", 0.0))

    schedule_kwargs = {}
    for key in ("beta_start", "beta_end", "cosine_s"):
        if key in merged:
            schedule_kwargs[key] = merged.pop(key)
    schedule = merged.pop("schedule_obj", None)
    if schedule is None:
        schedule = build_schedule(
            {"num_timesteps": num_timesteps, "schedule": schedule_name, "eta": eta, **schedule_kwargs}
        )

    # AN-specific values (paper defaults: J=10, omega=0.02)
    J = int(merged.pop("J", 10))
    omega = float(merged.pop("omega", 0.02))
    norm = merged.pop("norm", "per_sample")
    loss_reduction = merged.pop("loss_reduction", merged.pop("ant_loss_reduction", "sum"))
    grad_normalize = bool(merged.pop("grad_normalize", False))
    normalize_each_step = bool(merged.pop("normalize_each_step", True))
    clip_grad = merged.pop("clip_grad", None)
    clip_noise = merged.pop("clip_noise", None)
    detach_model_params = bool(merged.pop("detach_model_params", True))

    return AdversarialNoiseSelector(
        model=model,
        schedule=schedule,
        J=J,
        omega=omega,
        norm=norm,
        loss_reduction=loss_reduction,
        grad_normalize=grad_normalize,
        normalize_each_step=normalize_each_step,
        clip_grad=clip_grad,
        clip_noise=clip_noise,
        detach_model_params=detach_model_params,
    )
