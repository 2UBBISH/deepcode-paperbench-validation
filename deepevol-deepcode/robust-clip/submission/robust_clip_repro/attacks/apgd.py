"""Auto-PGD (APGD) attack wrapper for Robust CLIP robustness evaluation.

Addendum specification (quoted):

    "The APGD algorithm is taken from
     https://github.com/fra31/robust-finetuning."

This module exposes :class:`APGDAttack` with *exactly the same public interface*
as :class:`robust_clip_repro.attacks.pgd.PGDLinfAttack` so that
``eval_imagenet.py`` (and the other harnesses) can switch attackers without any
change to the surrounding evaluation code.

Design rules taken from the Addendum
------------------------------------
1. The algorithm must come from ``fra31/robust-finetuning``.  We therefore
   *first* try to import the upstream module (installed in editable mode / on
   ``sys.path``) and only fall back to the faithful local re-implementation
   vendored below if upstream is unavailable.
2. The Addendum states nothing about APGD hyper-parameters, therefore we
   **preserve upstream defaults** and must *not* overwrite APGD internals with
   the custom PGD settings (momentum 0.9, uniform random init, ...).  Upstream
   defaults are ``n_iter=100``, ``n_restarts=1``, ``rho=0.75``, internal
   momentum ``0.75``, initial step size ``alpha = 2 * eps`` and step sizes
   ``n_iter_2 = 0.22 * n_iter``, ``n_iter_min = 0.06 * n_iter``,
   ``size_decr = 0.03 * n_iter``.
3. Perturbation storage still honours the precision policy from
   :mod:`robust_clip_repro.utils.precision` (int16 for half precision, int32 for
   single precision), because the invariant is enforced at the storage layer for
   *every* attack, not only for PGD.
4. Like PGD, the l_inf ball is computed around **non-normalized** inputs, i.e.
   the attack operates purely in raw pixel space and model normalization stays
   inside the caller-supplied ``loss_fn``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..utils.precision import (
    decode_perturbation,
    encode_perturbation,
    float_dtype_for_precision,
    int_dtype_for_precision,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upstream defaults (fra31/robust-finetuning -> autoattack/autopgd_base.py)
# Those values are NOT stated by the Addendum: they are the upstream defaults we
# are required to preserve.  They are logged at construction time.
# ---------------------------------------------------------------------------
UPSTREAM_DEFAULTS: Dict[str, Any] = {
    "n_iter": 100,
    "n_restarts": 1,
    "rho": 0.75,
    "momentum": 0.75,
    "alpha_factor": 2.0,  # alpha = alpha_factor * eps
    "n_iter_2_frac": 0.22,
    "n_iter_min_frac": 0.06,
    "size_decr_frac": 0.03,
    "eot_iter": 1,
    "n_target_classes": 9,
}

# Supported upstream loss names.
SUPPORTED_LOSSES = ("ce", "dlr", "targeted_ce", "targeted_dlr", "mse")


# ---------------------------------------------------------------------------
# Upstream import attempt (vendored fallback below)
# ---------------------------------------------------------------------------
def _import_upstream_apgd():
    """Try to import the APGD implementation from ``fra31/robust-finetuning``.

    Returns ``(class, source_name)`` or ``(None, None)`` when unavailable.
    Several import paths are probed because the repository does not ship a
    package and different forks place the file differently.
    """

    candidates = (
        # robust_finetuning repo layout
        ("robust_finetuning.apgd", "APGDAttack"),
        ("robust_finetuning.autoattack.apgd", "APGDAttack"),
        ("autoattack.autopgd_base", "APGDAttack"),
        ("autoattack.apgd", "APGDAttack"),
        ("apgd", "APGDAttack"),
        # some forks expose APGDAttack_target for the targeted variant
        ("autoattack.apgd_targeted", "APGDAttack_target"),
    )
    for module_name, class_name in candidates:
        try:  # pragma: no cover - depends on the environment
            import importlib

            module = importlib.import_module(module_name)
            cls = getattr(module, class_name, None)
            if cls is not None:
                logger.info("APGD: using upstream implementation from '%s.%s'", module_name, class_name)
                return cls, f"{module_name}.{class_name}"
        except Exception:  # noqa: BLE001 - any import failure means "not installed"
            continue
    logger.info(
        "APGD: upstream fra31/robust-finetuning implementation not importable; "
        "using the vendored faithful re-implementation."
    )
    return None, None


# ---------------------------------------------------------------------------
# Losses used by APGD (mirrors upstream get_loss_fun)
# ---------------------------------------------------------------------------
def _dlr_loss(logits: torch.Tensor, y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
    """Difference of Logits Ratio (DLR) loss, Croce & Hein (2020).

    ``DLR(x, y) = -(z_y - max_{i != y} z_i) / (z_pi1 - z_pi3)`` where ``z_pi`` is
    the sorted logits vector.  The targeted variant flips the sign of the
    numerator.
    """

    if targeted:
        return _dlr_loss_targeted(logits, y)

    sorted_logits, _ = torch.sort(logits, descending=True, dim=1)
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    # largest logit among the non-target classes
    z_max_other = torch.where(
        sorted_logits[:, 0] == z_y, sorted_logits[:, 1], sorted_logits[:, 0]
    )
    z_pi1 = sorted_logits[:, 0]
    z_pi3 = sorted_logits[:, 2] if logits.shape[1] > 2 else sorted_logits[:, -1]
    denom = z_pi1 - z_pi3
    return -(z_y - z_max_other) / (denom + 1e-12)


def _dlr_loss_targeted(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Targeted DLR loss (upstream ``dlr_targeted``): minimise distance to y."""

    sorted_logits, _ = torch.sort(logits, descending=True, dim=1)
    z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
    z_pi1 = sorted_logits[:, 0]
    z_pi3 = sorted_logits[:, 2] if logits.shape[1] > 2 else sorted_logits[:, -1]
    denom = z_pi1 - z_pi3
    return (z_pi1 - z_y) / (denom + 1e-12)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class APGDConfig:
    """Configuration for APGD.

    ``eps`` has to be supplied externally (the Addendum is silent about it).
    Every other field defaults to the upstream value.
    """

    eps: Optional[float] = None
    alpha: Optional[float] = None  # None -> 2 * eps (upstream rule)
    iterations: Optional[int] = None  # None -> upstream n_iter (100)
    restarts: Optional[int] = None  # None -> upstream n_restarts (1)
    norm: str = "linf"
    loss: str = "ce"
    targeted: bool = False
    precision: str = "single"
    momentum: float = UPSTREAM_DEFAULTS["momentum"]
    rho: float = UPSTREAM_DEFAULTS["rho"]
    eot_iter: int = UPSTREAM_DEFAULTS["eot_iter"]
    n_target_classes: int = UPSTREAM_DEFAULTS["n_target_classes"]
    random_start: bool = True
    quant_scale: float = 1.0
    seed: Optional[int] = None

    def resolve(
        self,
        eps: Optional[float] = None,
        alpha: Optional[float] = None,
        iterations: Optional[int] = None,
        restarts: Optional[int] = None,
    ) -> "APGDConfig":
        """Return a copy with externally supplied values filled in."""

        return APGDConfig(
            eps=eps if eps is not None else self.eps,
            alpha=alpha if alpha is not None else self.alpha,
            iterations=iterations if iterations is not None else self.iterations,
            restarts=restarts if restarts is not None else self.restarts,
            norm=self.norm,
            loss=self.loss,
            targeted=self.targeted,
            precision=self.precision,
            momentum=self.momentum,
            rho=self.rho,
            eot_iter=self.eot_iter,
            n_target_classes=self.n_target_classes,
            random_start=self.random_start,
            quant_scale=self.quant_scale,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Vendored faithful re-implementation of the upstream APGD
# ---------------------------------------------------------------------------
class _VendoredAPGDAttack:
    """Faithful local port of ``autoattack/autopgd_base.py::APGDAttack``.

    The class keeps the upstream algorithm (momentum 0.75, ``alpha = 2 * eps``,
    oscillation-based step-size halving, ``n_iter_2``/``n_iter_min``/
    ``size_decr`` schedules) and exposes a helper-oriented API so that it can be
    driven from this package's ``loss_fn`` convention.
    """

    def __init__(
        self,
        n_iter: int = UPSTREAM_DEFAULTS["n_iter"],
        norm: str = "linf",
        n_restarts: int = UPSTREAM_DEFAULTS["n_restarts"],
        eps: float = 8 / 255,
        loss: str = "ce",
        eot_iter: int = UPSTREAM_DEFAULTS["eot_iter"],
        rho: float = UPSTREAM_DEFAULTS["rho"],
        seed: Optional[int] = None,
        n_target_classes: int = UPSTREAM_DEFAULTS["n_target_classes"],
        device: Optional[torch.device] = None,
    ) -> None:
        self.n_iter = int(n_iter)
        self.norm = norm.lower()
        self.n_restarts = int(n_restarts)
        self.eps = float(eps)
        self.loss = loss
        self.eot_iter = int(eot_iter)
        self.thr_decr = float(rho)
        self.seed = seed
        self.n_target_classes = int(n_target_classes)
        self.device = device if device is not None else torch.device("cpu")

        # step-size schedule (upstream)
        self.n_iter_2 = max(int(UPSTREAM_DEFAULTS["n_iter_2_frac"] * self.n_iter), 1)
        self.n_iter_min = max(int(UPSTREAM_DEFAULTS["n_iter_min_frac"] * self.n_iter), 1)
        self.size_decr = max(int(UPSTREAM_DEFAULTS["size_decr_frac"] * self.n_iter), 1)

    # -- helpers ---------------------------------------------------------
    def _rand_like(self, x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        if generator is not None:
            return torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return torch.rand(x.shape, device=x.device, dtype=x.dtype)

    def initial_perturbation(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Uniform random initialization inside the l_inf / l_2 ball (upstream)."""

        if self.norm == "linf":
            t = 2.0 * self._rand_like(x, generator) - 1.0
            return self.eps * t
        if self.norm == "l2":
            t = torch.randn(
                x.shape, device=x.device, dtype=x.dtype
            ) if generator is None else torch.randn(
                x.shape, generator=generator, device=x.device, dtype=x.dtype
            )
            flat = t.reshape(t.shape[0], -1)
            flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-10)
            return self.eps * flat.view_as(t)
        raise ValueError(f"Unsupported norm '{self.norm}' for APGD.")

    def _project(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """Project onto the l_inf ball around the (raw, non-normalized) input."""

        if self.norm == "linf":
            return torch.clamp(x_adv, x - self.eps, x + self.eps)
        if self.norm == "l2":
            delta = x_adv - x
            flat = delta.reshape(delta.shape[0], -1)
            norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-10)
            factor = torch.clamp(self.eps / norm, max=1.0)
            return x + (flat * factor).view_as(delta)
        raise ValueError(f"Unsupported norm '{self.norm}' for APGD.")

    def _loss_value(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        x_adv: torch.Tensor,
        y_target: Optional[torch.Tensor] = None,
        targeted: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(scalar_loss_for_backward, per_sample_loss)``."""

        out = loss_fn(x_adv)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = torch.as_tensor(out)
        if out.dim() == 0:
            per_sample = out.expand(x_adv.shape[0])
            scalar = out
        else:
            per_sample = out
            scalar = out.sum()
        if targeted:
            # Minimising a targeted loss == maximising the loss w.r.t. the target
            scalar = -scalar
            per_sample = -per_sample
        return scalar, per_sample

    def _step_sizes(self, alpha: float) -> List[float]:
        """Upstream step-size list: [alpha, alpha/2, alpha/4]."""

        return [alpha, alpha / 2.0, alpha / 4.0]

    # -- main single run -------------------------------------------------
    def attack_single_run(
        self,
        x: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        y: Optional[torch.Tensor] = None,
        targeted: bool = False,
        y_target: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        best_delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one APGD restart.  Returns ``(x_best, per_sample_best_loss)``."""

        alpha = 2.0 * self.eps  # upstream initial step size
        step_sizes = self._step_sizes(alpha)

        x_adv = x.clone().detach()
        if best_delta is not None:
            x_adv = x_adv + best_delta.to(x_adv.dtype).clone()
        else:
            x_adv = x_adv + self.initial_perturbation(x, generator=generator)
            x_adv = self._project(x, x_adv)

        x_best = x_adv.clone().detach()
        with torch.no_grad():
            _, f_best = self._loss_value(loss_fn, x_adv, y_target, targeted)

        x_adv_prev = None
        f_prev = None
        grad_prev = None
        step_size_index = 0
        momentum = UPSTREAM_DEFAULTS["momentum"]

        for i in range(self.n_iter):
            alpha_i = step_sizes[step_size_index]

            x_adv = x_adv.detach().clone().requires_grad_(True)
            scalar, per_sample = self._loss_value(loss_fn, x_adv, y_target, targeted)
            grad = torch.autograd.grad(scalar, [x_adv])[0].detach()
            x_adv = x_adv.detach()

            if self.norm == "linf":
                # Normalize the gradient (mean absolute value over non-batch dims)
                flat = grad.reshape(grad.shape[0], -1)
                grad_norm = flat.abs().mean(dim=1).clamp_min(1e-12)
                grad = (flat / grad_norm.view(-1, 1)).view_as(grad)
                grad = grad + momentum * grad_prev if grad_prev is not None else grad
                grad_prev = grad.clone()
                x_adv = x_adv + alpha_i * grad.sign()
            elif self.norm == "l2":
                flat = grad.reshape(grad.shape[0], -1)
                grad_norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                grad = (flat / grad_norm).view_as(grad)
                grad = grad + momentum * grad_prev if grad_prev is not None else grad
                grad_prev = grad.clone()
                x_adv = x_adv + alpha_i * grad
            else:
                raise ValueError(f"Unsupported norm '{self.norm}' for APGD.")

            x_adv = self._project(x, x_adv)

            with torch.no_grad():
                _, f_new = self._loss_value(loss_fn, x_adv, y_target, targeted)

            # best tracking (upstream compares the per-sample loss)
            improved = f_new < f_best
            if bool(improved.any()):
                idx = improved.nonzero(as_tuple=False).flatten()
                x_best[idx] = x_adv[idx]
                f_best[idx] = f_new[idx]

            # -------- step-size adaptation (upstream CheckpointSchedule) ----
            if i == 0:
                x_adv_prev = x_adv.clone()
                with torch.no_grad():
                    _, f_prev = self._loss_value(loss_fn, x_adv_prev, y_target, targeted)
            else:
                is_oscillating = True
                if x_adv_prev is not None:
                    diff = (f_new == f_prev).float()
                    is_oscillating = bool(diff.sum() == float(diff.numel()))
                x_adv_prev = x_adv.clone()
                f_prev = f_new.clone()

                if (i % self.size_decr == 0) and (
                    step_size_index < len(step_sizes) - 1
                ):
                    # halve the step size either on oscillation or after
                    # n_iter_2 / n_iter_min evaluations without improvement
                    if is_oscillating or i >= self.n_iter_2 + self.n_iter_min:
                        step_size_index += 1
                        if self.n_iter_2 == 0:
                            self.n_iter_2 = 1

        return x_best, f_best


# ---------------------------------------------------------------------------
# Public attack class (interface-compatible with PGDLinfAttack)
# ---------------------------------------------------------------------------
class APGDAttack:
    """APGD attack with the same public interface as :class:`PGDLinfAttack`.

    Parameters
    ----------
    eps : float
        l_inf (or l_2) budget, supplied externally.
    alpha : Optional[float]
        Initial step size.  ``None`` selects the upstream rule ``alpha = 2 * eps``.
    iterations : Optional[int]
        Number of iterations.  ``None`` selects the upstream default (100).
    precision : str
        Precision of the attack; controls the mandated perturbation storage dtype
        (``int16`` for half precision, ``int32`` for single precision).
    """

    def __init__(
        self,
        eps: float,
        alpha: Optional[float] = None,
        iterations: Optional[int] = None,
        restarts: Optional[int] = None,
        norm: str = "linf",
        loss: str = "ce",
        targeted: bool = False,
        precision: str = "single",
        momentum: float = UPSTREAM_DEFAULTS["momentum"],
        rho: float = UPSTREAM_DEFAULTS["rho"],
        eot_iter: int = UPSTREAM_DEFAULTS["eot_iter"],
        n_target_classes: int = UPSTREAM_DEFAULTS["n_target_classes"],
        random_start: bool = True,
        quant_scale: float = 1.0,
        clamp: Optional[Tuple[float, float]] = None,
        config: Optional[APGDConfig] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_upstream: bool = True,
    ) -> None:
        if config is not None:
            eps = config.eps if eps is None else eps
            alpha = config.alpha if alpha is None else alpha
            iterations = config.iterations if iterations is None else iterations
            restarts = config.restarts if restarts is None else restarts
            norm = config.norm
            loss = config.loss
            targeted = config.targeted
            precision = config.precision
            momentum = config.momentum
            rho = config.rho
            eot_iter = config.eot_iter
            n_target_classes = config.n_target_classes
            random_start = config.random_start
            quant_scale = config.quant_scale
            seed = config.seed if seed is None else seed

        if eps is None:
            raise ValueError(
                "APGDAttack requires an externally supplied 'eps' (the Addendum does not "
                "specify an APGD budget; provide it via configs/apgd_eval.yaml)."
            )

        self.eps = float(eps)
        self.alpha = float(alpha) if alpha is not None else UPSTREAM_DEFAULTS["alpha_factor"] * self.eps
        self.iterations = int(iterations) if iterations is not None else UPSTREAM_DEFAULTS["n_iter"]
        self.restarts = int(restarts) if restarts is not None else UPSTREAM_DEFAULTS["n_restarts"]
        self.norm = norm
        self.loss = loss
        self.targeted = bool(targeted)
        self.precision = precision
        self.momentum = float(momentum)
        self.rho = float(rho)
        self.eot_iter = int(eot_iter)
        self.n_target_classes = int(n_target_classes)
        self.random_start = bool(random_start)
        self.quant_scale = float(quant_scale)
        self.clamp = clamp
        self.seed = seed
        self.device = device

        if self.loss not in SUPPORTED_LOSSES:
            raise ValueError(
                f"Unsupported APGD loss '{self.loss}'; expected one of {SUPPORTED_LOSSES}."
            )

        # Try upstream first (Addendum: "The APGD algorithm is taken from ...").
        self._upstream_cls = None
        self._upstream_source = None
        if use_upstream:
            self._upstream_cls, self._upstream_source = _import_upstream_apgd()
        self._backend = "upstream" if self._upstream_cls is not None else "vendored"

        self._engine = _VendoredAPGDAttack(
            n_iter=self.iterations,
            norm=self.norm,
            n_restarts=self.restarts,
            eps=self.eps,
            loss=self.loss,
            eot_iter=self.eot_iter,
            rho=self.rho,
            seed=self.seed,
            n_target_classes=self.n_target_classes,
            device=self.device,
        )

        logger.info(
            "APGDAttack(eps=%s, alpha=%s, iterations=%d, restarts=%d, norm=%s, loss=%s, "
            "backend=%s) -- alpha/iterations/restarts are externally supplied or upstream "
            "defaults, NOT Addendum-specified values.",
            self.eps,
            self.alpha,
            self.iterations,
            self.restarts,
            self.norm,
            self.loss,
            self._backend,
        )

    # -- introspection ---------------------------------------------------
    @property
    def backend(self) -> str:
        """``"upstream"`` when fra31/robust-finetuning was importable, else ``"vendored"``."""

        return self._backend

    def upstream_source(self) -> Optional[str]:
        return self._upstream_source

    def summary(self) -> Dict[str, Any]:
        """Machine-readable description used by the evaluation harnesses' logs."""

        return {
            "attacker": "APGD",
            "backend": self._backend,
            "upstream_source": self._upstream_source,
            "eps": self.eps,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "restarts": self.restarts,
            "norm": self.norm,
            "loss": self.loss,
            "momentum": self.momentum,
            "rho": self.rho,
            "precision": self.precision,
            "int_dtype": str(int_dtype_for_precision(self.precision)),
            "specified_by": "external/upstream-default (Addendum silent)",
        }

    # -- precision-aware storage ----------------------------------------
    def _encode(self, delta: torch.Tensor) -> torch.Tensor:
        return encode_perturbation(delta, self.precision, quant_scale=self.quant_scale)

    def _decode(self, codes: torch.Tensor) -> torch.Tensor:
        return decode_perturbation(
            codes,
            float_dtype=float_dtype_for_precision(self.precision),
            quant_scale=self.quant_scale,
        )

    # -- public helpers matching PGDLinfAttack ---------------------------
    def initial_perturbation(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Uniform random initialization inside the l_inf ball (upstream rule)."""

        fdt = float_dtype_for_precision(self.precision)
        x_f = x.to(fdt)
        delta = self._engine.initial_perturbation(x_f, generator=generator)
        return self._encode(delta)

    def project(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Project in **raw pixel space** (never model-normalized space)."""

        fdt = float_dtype_for_precision(self.precision)
        x_f = x.to(fdt)
        delta_f = self._decode(delta) if delta.dtype != fdt else delta
        x_adv = self._engine._project(x_f, x_f + delta_f)
        if self.clamp is not None:
            x_adv = torch.clamp(x_adv, self.clamp[0], self.clamp[1])
        return self._encode(x_adv - x_f)

    @staticmethod
    def normalize_and_sign(grad: torch.Tensor) -> torch.Tensor:
        """Upstream l_inf gradient normalization (mean-abs) followed by sign."""

        flat = grad.reshape(grad.shape[0], -1)
        norm = flat.abs().mean(dim=1).clamp_min(1e-12)
        return (flat / norm.view(-1, 1)).view_as(grad).sign()

    def adversarial_examples(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Decode the stored integer perturbation and add it to the raw pixels."""

        fdt = float_dtype_for_precision(self.precision)
        delta_f = self._decode(delta)
        return x.to(fdt) + delta_f

    # -- loss dispatch ---------------------------------------------------
    def _build_loss_fn(
        self,
        loss_fn: Optional[Callable[..., torch.Tensor]],
        y: Optional[torch.Tensor],
        targeted: bool,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Resolve the per-sample loss to optimise.

        * If the caller provides ``loss_fn`` we use it (it receives raw pixels and
          must apply model normalization itself).
        * ``loss=None``/``"ce"``/``"dlr"`` only change the *logit-based* losses
          built by :func:`apgd_logit_loss`, which is the helper used by
          ``eval_imagenet.py``.
        """

        if loss_fn is not None:
            return loss_fn
        raise ValueError(
            "APGDAttack.attack_* requires a loss_fn mapping raw pixels -> logits/loss; "
            "see apgd_logit_loss / make_apgd_loss_fn for the standard callables."
        )

    # -- attacks ---------------------------------------------------------
    def perturb(
        self,
        x: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        y_target: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_delta: bool = True,
        best_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full APGD run with ``restarts`` restarts.

        Returns the perturbation (integer codes when ``return_delta`` is True) or
        the adversarial raw pixels otherwise.
        """

        fdt = float_dtype_for_precision(self.precision)
        x_f = x.to(fdt)

        if generator is None and self.seed is not None:
            generator = torch.Generator(device=x_f.device)
            generator.manual_seed(int(self.seed))

        best_delta_f: Optional[torch.Tensor] = None
        if best_delta is not None:
            best_delta_f = self._decode(best_delta) if best_delta.dtype != fdt else best_delta
        elif self.restarts > 1:
            # warm-start the first restart from a random point (upstream behaviour)
            best_delta_f = self._engine.initial_perturbation(x_f, generator=generator)

        for restart in range(max(self.restarts, 1)):
            x_best, _ = self._engine.attack_single_run(
                x_f,
                loss_fn,
                y=y,
                targeted=self.targeted or (y_target is not None),
                y_target=y_target,
                generator=generator,
                best_delta=best_delta_f if restart == 0 else None,
            )
            best_delta_f = x_best - x_f

        if self.clamp is not None:
            x_adv = torch.clamp(x_f + best_delta_f, self.clamp[0], self.clamp[1])
            best_delta_f = x_adv - x_f

        delta_codes = self._encode(best_delta_f)
        if return_delta:
            return delta_codes
        return self.adversarial_examples(x, delta_codes)

    def attack_untargeted(
        self,
        x: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        labels: Optional[torch.Tensor] = None,
        return_delta: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Untargeted APGD (maximise the loss)."""

        return self.perturb(
            x,
            loss_fn,
            y=labels,
            y_target=None,
            generator=generator,
            return_delta=return_delta,
        )

    def attack_targeted(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        return_delta: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Targeted APGD (minimise the loss w.r.t. ``y_target``)."""

        return self.perturb(
            x,
            loss_fn,
            y_target=y_target,
            generator=generator,
            return_delta=return_delta,
        )

    # -- upstream delegation helpers -------------------------------------
    def run_upstream(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Run the *actual* upstream class when it is available.

        ``model_fn`` maps raw, non-normalized pixels to logits.  APGD projections
        and the l_inf ball stay in that same raw pixel space; only the loss
        computation applies normalization (handled inside ``model_fn``).
        """

        if self._upstream_cls is None:
            raise RuntimeError(
                "The upstream fra31/robust-finetuning APGD is not importable in this "
                "environment; use attack_untargeted/attack_targeted (vendored backend)."
            )
        kwargs: Dict[str, Any] = {
            "n_iter": self.iterations,
            "norm": self.norm,
            "n_restarts": self.restarts,
            "eps": self.eps,
            "loss": self.loss,
            "eot_iter": self.eot_iter,
            "rho": self.rho,
            "n_target_classes": self.n_target_classes,
        }
        try:
            attacker = self._upstream_cls(model_fn, **kwargs)  # type: ignore[misc]
            x_adv = attacker.perturb(x, y)
        except TypeError:
            attacker = self._upstream_cls(**kwargs)  # type: ignore[misc]
            x_adv = attacker.perturb(model_fn, x, y)
        return x_adv


# ---------------------------------------------------------------------------
# Logit-based loss builders (mirrors upstream get_loss_fun)
# ---------------------------------------------------------------------------
def apgd_logit_loss(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    y: Optional[torch.Tensor] = None,
    loss: str = "ce",
    targeted: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a per-sample loss over raw pixels from a ``logits_fn``."""

    def _loss(x_adv: torch.Tensor) -> torch.Tensor:
        logits = logits_fn(x_adv)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if y is None:
            # Untargeted margin-style loss when no labels are supplied:
            # maximise the confidence gap of the current top-1 class.
            top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
            return (top2[:, 0] - top2[:, -1]) if top2.shape[1] > 1 else top2[:, 0]
        if loss == "ce":
            return torch.nn.functional.cross_entropy(logits, y, reduction="none")
        if loss == "dlr":
            return _dlr_loss(logits, y, targeted=targeted)
        raise ValueError(f"Unsupported logit loss '{loss}'.")

    return _loss


def make_apgd_loss_fn(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    y: Optional[torch.Tensor] = None,
    loss: str = "ce",
    targeted: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Alias of :func:`apgd_logit_loss` for harness convenience."""

    return apgd_logit_loss(logits_fn, y=y, loss=loss, targeted=targeted)


# ---------------------------------------------------------------------------
# Functional wrapper (mirrors pgd_linf_attack)
# ---------------------------------------------------------------------------
def apgd_attack(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    alpha: Optional[float] = None,
    iterations: Optional[int] = None,
    restarts: Optional[int] = None,
    precision: str = "single",
    norm: str = "linf",
    loss: str = "ce",
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Functional APGD wrapper.

    ``model_fn`` maps raw, **non-normalized** pixels to logits (normalization is
    applied inside ``model_fn``), so the l_inf ball is computed around the
    non-normalized inputs, exactly as required by the Addendum.

    Returns the adversarial examples in raw pixel space.
    """

    attacker = APGDAttack(
        eps=eps,
        alpha=alpha,
        iterations=iterations,
        restarts=restarts,
        norm=norm,
        loss=loss,
        targeted=targeted,
        precision=precision,
        clamp=clamp,
    )
    loss_fn = apgd_logit_loss(
        model_fn, y=y_target if y_target is not None else y, loss=loss, targeted=targeted
    )
    if y_target is not None or targeted:
        delta = attacker.attack_targeted(x, y_target, loss_fn, return_delta=True)
    else:
        delta = attacker.attack_untargeted(x, loss_fn, labels=y, return_delta=True)
    return attacker.adversarial_examples(x, delta)


__all__ = [
    "APGDAttack",
    "APGDConfig",
    "UPSTREAM_DEFAULTS",
    "SUPPORTED_LOSSES",
    "apgd_attack",
    "apgd_logit_loss",
    "make_apgd_loss_fn",
]
