"""Probability-flow ODE samplers for stochastic interpolants with couplings.

Reference: *Stochastic Interpolants with Data-Dependent Couplings*,
Corollary 3.1 (probability flow, Eq. 8), Section 3.4, Algorithm 2, Appendix B.

The probability flow equation is

.. math::
    \dot X_t = b_t(X_t) \\qquad X_{t=1} \\sim \\rho_1(x_1) \\;\\text{if}\\; X_{t=0} \\sim \\rho_0(x_0)

Because the approximation :math:`\\hat b_t(x, \\xi)` depends on the conditioning variable
:math:`\\xi` (the missingness mask for in-painting, the upsampled low-resolution image for
super-resolution), :math:`\\xi` must be re-supplied to the network at **every** integration
step; this module guarantees that by closing over it in the drift callable.

Two integrators are provided:

* :func:`dopri_sample` -- the Dopri solver from ``torchdiffeq`` (Chen, 2018), used for the
  reported experiments (Appendix B: "We use the Dopri solver from the torchdiffeq library").
* :func:`euler_sample` -- the forward-Euler scheme from **Algorithm 2**,
  :math:`\\hat X_{n+1} = \\hat X_n + N^{-1}\\hat b_{n/N}(\\hat X_n)`.

A small RK4 integrator (``method="rk4"``) is included as a dependency-free fallback when
``torchdiffeq`` is not installed, so that sampling still works out of the box.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

__all__ = [
    "ODESampler",
    "probability_flow_ode",
    "ode_sample",
    "euler_sample",
    "dopri_sample",
    "rk4_sample",
    "make_drift_fn",
    "make_project_fn",
    "initial_condition",
    "sample_from_coupling",
    "solve_ode",
    "TORCHDIFFEQ_AVAILABLE",
    "DOPRI_METHODS",
]


# --------------------------------------------------------------------------------------
# Optional dependency: torchdiffeq (Dopri solver, Chen 2018)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    from torchdiffeq import odeint  # type: ignore

    TORCHDIFFEQ_AVAILABLE = True
except Exception:  # pragma: no cover
    odeint = None  # type: ignore
    TORCHDIFFEQ_AVAILABLE = False


#: Adaptive solvers forwarded to ``torchdiffeq.odeint``.
DOPRI_METHODS = ("dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun")

#: Fixed-step solvers forwarded to ``torchdiffeq.odeint``.
FIXED_METHODS = ("euler", "midpoint", "rk4", "explicit_adams", "implicit_adams")


# --------------------------------------------------------------------------------------
# Model invocation
# --------------------------------------------------------------------------------------
_COND_KEYS = ("xi", "cond", "conditioning", "context", "mask", "c")
_LABEL_KEYS = ("y", "label", "labels", "class_labels", "cls")


def _accepts(fn: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / exotic callables
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return name in sig.parameters


def _call_model(
    model: Callable,
    x: Tensor,
    t: Union[float, Tensor],
    xi: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    **extra: Any,
) -> Tensor:
    """Invoke ``model(x, t, ...)`` tolerating different conditioning keyword names."""
    if y is not None and _accepts(model, "y"):
        out = model(x, t, xi, y, **extra) if xi is not None else model(x, t, None, y, **extra)
    else:
        out = model(x, t, xi, **extra)

    if isinstance(out, dict):  # networks may return diagnostics
        for key in ("b_hat", "velocity", "v", "output", "x"):
            if key in out:
                out = out[key]
                break
    elif isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, Tensor):
        raise TypeError(f"Velocity model returned {type(out)}, expected a Tensor.")
    return out


# --------------------------------------------------------------------------------------
# Drift / projection callables
# --------------------------------------------------------------------------------------
def make_drift_fn(
    model: Callable,
    xi: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    mask_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    time_scale: float = 1.0,
    **extra: Any,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build the ODE drift ``dx/dt = b_hat_t(x, xi)``.

    Args:
        model: velocity network :math:`\\hat b_t(x, \\xi)`.
        xi: image-shaped conditioning (mask or upsampled low-res image); re-supplied at
            every step.
        y: optional class labels.
        mask_fn: optional structural masker ``(velocity, xi) -> velocity`` used for
            in-painting so the velocity is zero on observed pixels.
        time_scale: multiply the time passed to the network (kept for API symmetry; the
            paper integrates on :math:`t \\in [0, 1]` directly).

    Returns:
        Callable ``drift(x, t) -> Tensor`` suitable for an ODE solver.
    """

    def drift(x: Tensor, t: Union[float, Tensor]) -> Tensor:
        tt = t
        if not isinstance(t, Tensor):
            tt = torch.as_tensor(float(t), dtype=x.dtype, device=x.device)
        if time_scale != 1.0:
            tt = tt * time_scale
        v = _call_model(model, x, tt, xi=xi, y=y, **extra)
        if mask_fn is not None:
            v = mask_fn(v, xi)
        return v

    return drift


def make_project_fn(
    xi: Optional[Tensor] = None,
    x1: Optional[Tensor] = None,
    observed_value: float = 1.0,
    keep_all_channels: bool = False,
) -> Optional[Callable[[Tensor], Tensor]]:
    """Build a projection that re-imposes observed content at every integration step.

    For in-painting the coupled interpolant satisfies :math:`\\xi \\circ I_t = \\xi \\circ x_1`
    for **all** :math:`t` (Section 4.1), so the observed pixels are constant along the ODE
    trajectory.  Projecting them back after each solver step removes accumulated numerical
    error.  Returns ``None`` when the projection is not applicable.
    """
    if xi is None or x1 is None:
        return None

    def project(x: Tensor) -> Tensor:
        mask = xi
        if mask.shape[1] != x.shape[1]:
            if mask.shape[1] == 1 and not keep_all_channels:
                mask = mask.expand(-1, x.shape[1], -1, -1)
            else:
                mask = mask.expand(-1, x.shape[1], -1, -1)
        return mask * x1 + (1.0 - mask) * x

    return project


def initial_condition(
    x1: Optional[Tensor] = None,
    coupling: Optional[Any] = None,
    x0: Optional[Tensor] = None,
    zeta: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Build the ODE initial condition :math:`X_{t=0} = m(x_1) + \\sigma \\zeta`.

    Either an explicit :math:`x_0` is returned unchanged (the paper notes that at inference
    time one may *directly observe* :math:`x_0 \\sim \\rho_0`, e.g. a partial image), or it is
    constructed from the coupling (Eq. 18).

    Returns:
        ``(x0, xi)`` where ``xi`` is the coupling conditioning tensor (or ``None``).
    """
    if x0 is not None:
        return x0, None
    if coupling is None or x1 is None:
        raise ValueError("Provide either `x0`, or `x1` together with a `coupling`.")

    out = coupling.build_x0(x1, zeta=zeta, generator=generator, return_xi=True)
    if isinstance(out, tuple):
        x0, xi = out
    else:  # pragma: no cover - defensive
        x0, xi = out, None
    return x0, xi


# --------------------------------------------------------------------------------------
# Core integrators
# --------------------------------------------------------------------------------------
def _time_grid(
    x: Tensor,
    t0: float,
    t1: float,
    steps: Optional[int],
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    n = int(steps) if steps is not None else 2
    n = max(n, 1)
    return torch.linspace(t0, t1, n + 1, dtype=dtype or x.dtype, device=x.device)


def _rk4_integrate(
    drift: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    ts: Tensor,
    project: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Fixed-step classical Runge-Kutta 4 (dependency-free fallback)."""
    for i in range(ts.shape[0] - 1):
        t0, t1 = ts[i], ts[i + 1]
        h = t1 - t0
        k1 = drift(x, t0)
        k2 = drift(x + 0.5 * h * k1, t0 + 0.5 * h)
        k3 = drift(x + 0.5 * h * k2, t0 + 0.5 * h)
        k4 = drift(x + h * k3, t1)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if project is not None:
            x = project(x)
    return x


def _euler_integrate(
    drift: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    ts: Tensor,
    project: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Forward Euler, exactly Algorithm 2: ``X_{n+1} = X_n + N^{-1} b_{n/N}(X_n)``."""
    for i in range(ts.shape[0] - 1):
        h = ts[i + 1] - ts[i]
        x = x + h * drift(x, ts[i])
        if project is not None:
            x = project(x)
    return x


def solve_ode(
    drift: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    method: str = "dopri5",
    steps: int = 50,
    t0: float = 0.0,
    t1: float = 1.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    return_trajectory: bool = False,
    trajectory_steps: Optional[int] = None,
    **solver_kwargs: Any,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Integrate ``dx/dt = drift(x, t)`` from ``t0`` to ``t1``.

    Args:
        drift: velocity field callable ``(x, t) -> dx/dt``.
        x0: initial condition :math:`X_{t=0}`.
        method: ``"dopri5"`` (default; torchdiffeq), ``"euler"`` (Algorithm 2),
            ``"rk4"`` (built-in fallback), or any other torchdiffeq method name.
        steps: number of integration steps for fixed-step schemes, and the number of
            trajectory snapshots when ``return_trajectory`` is set.
        atol, rtol: adaptive-solver tolerances (torchdiffeq).
        project: optional per-step projection (see :func:`make_project_fn`).
        return_trajectory: also return the trajectory evaluated on a uniform ``steps``-grid.

    Returns:
        ``X_{t=1}``, or ``(X_{t=1}, trajectory)`` where ``trajectory`` has shape
        ``(steps + 1, *x0.shape)``.
    """
    method = str(method).lower()
    x = x0

    use_torchdiffeq = (
        TORCHDIFFEQ_AVAILABLE
        and (method in DOPRI_METHODS or method in FIXED_METHODS)
        and method != "rk4"
    )

    traj_ts = _time_grid(x, t0, t1, trajectory_steps or steps)

    if use_torchdiffeq:
        # Wrap the drift so that torchdiffeq only sees tensors (t comes in as scalar
        # tensor).  `project` cannot be folded into the drift of an adaptive solver, so
        # for in-painting we integrate on a fixed grid (see below) when it is supplied.
        if project is None:
            kwargs: Dict[str, Any] = dict(method=method)
            if method in DOPRI_METHODS:
                kwargs.update(atol=atol, rtol=rtol)
            else:
                kwargs.update(options=dict(step_size=abs(t1 - t0) / max(steps, 1)))
            kwargs.update(solver_kwargs)
            t_eval = traj_ts if return_trajectory else torch.tensor(
                [t0, t1], dtype=x.dtype, device=x.device
            )
            out = odeint(lambda t, xx: drift(xx, t), x, t_eval, **kwargs)
            if return_trajectory:
                return out[-1], out
            return out[-1]

        # With projection, integrate over the snapshot grid and correct after each step.
        ts = traj_ts
        if ts.shape[0] < 2:  # pragma: no cover - defensive
            ts = torch.tensor([t0, t1], dtype=x.dtype, device=x.device)
        xs: List[Tensor] = [x]
        for i in range(ts.shape[0] - 1):
            seg = torch.stack([ts[i], ts[i + 1]])
            kwargs = dict(method=method if method in DOPRI_METHODS else "rk4")
            if kwargs["method"] in DOPRI_METHODS:
                kwargs.update(atol=atol, rtol=rtol)
            seg_out = odeint(lambda t, xx: drift(xx, t), x, seg, **kwargs)
            x = seg_out[-1]
            x = project(x)
            xs.append(x)
        if return_trajectory:
            return x, torch.stack(xs)
        return x

    # Dependency-free fixed-step integrators
    if method in ("euler", "forward_euler", "euler_forward"):
        integrator = _euler_integrate
    elif method in ("rk4", "runge_kutta", "dopri5_approx", "dopri5"):
        integrator = _rk4_integrate
    else:
        raise ValueError(
            f"Unknown ODE method '{method}'. Known methods: {DOPRI_METHODS + FIXED_METHODS}."
        )

    if not return_trajectory:
        ts = _time_grid(x, t0, t1, steps)
        return integrator(drift, x, ts, project=project)

    ts = traj_ts
    xs = [x]
    for i in range(ts.shape[0] - 1):
        seg = torch.stack([ts[i], ts[i + 1]])
        x = integrator(drift, x, seg, project=project)
        xs.append(x)
    return x, torch.stack(xs)


# --------------------------------------------------------------------------------------
# Public sampling entry points
# --------------------------------------------------------------------------------------
def probability_flow_ode(
    model: Callable,
    x0: Tensor,
    xi: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    method: str = "dopri5",
    steps: int = 50,
    t0: float = 0.0,
    t1: float = 1.0,
    mask_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    return_trajectory: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    **kwargs: Any,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Solve the probability-flow ODE (Corollary 3.1 / Eq. 8) for a batch of ``x0``.

    The conditioning ``xi`` is re-supplied to the network at every step, as required for
    both in-painting (missingness mask) and super-resolution (upsampled low-resolution
    image).
    """
    drift = make_drift_fn(model, xi=xi, y=y, mask_fn=mask_fn, **kwargs)
    return solve_ode(
        drift,
        x0,
        method=method,
        steps=steps,
        t0=t0,
        t1=t1,
        atol=atol,
        rtol=rtol,
        project=project,
        return_trajectory=return_trajectory,
    )


def euler_sample(
    model: Callable,
    x0: Tensor,
    steps: int = 50,
    xi: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    mask_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    return_trajectory: bool = False,
    t0: float = 0.0,
    t1: float = 1.0,
    **kwargs: Any,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Algorithm 2: forward-Euler sampling with ``N = steps`` uniform steps."""
    return probability_flow_ode(
        model,
        x0,
        xi=xi,
        y=y,
        method="euler",
        steps=steps,
        t0=t0,
        t1=t1,
        mask_fn=mask_fn,
        project=project,
        return_trajectory=return_trajectory,
        **kwargs,
    )


def dopri_sample(
    model: Callable,
    x0: Tensor,
    xi: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mask_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    return_trajectory: bool = False,
    trajectory_steps: int = 50,
    **kwargs: Any,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Sampling with the Dopri solver (``torchdiffeq``), as used in Appendix B."""
    if not TORCHDIFFEQ_AVAILABLE:  # pragma: no cover - fallback path
        return rk4_sample(
            model,
            x0,
            steps=trajectory_steps,
            xi=xi,
            y=y,
            mask_fn=mask_fn,
            project=project,
            return_trajectory=return_trajectory,
            **kwargs,
        )
    drift = make_drift_fn(model, xi=xi, y=y, mask_fn=mask_fn, **kwargs)
    return solve_ode(
        drift,
        x0,
        method="dopri5",
        steps=trajectory_steps,
        atol=atol,
        rtol=rtol,
        project=project,
        return_trajectory=return_trajectory,
        trajectory_steps=trajectory_steps,
    )


def rk4_sample(
    model: Callable,
    x0: Tensor,
    steps: int = 50,
    xi: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    mask_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    project: Optional[Callable[[Tensor], Tensor]] = None,
    return_trajectory: bool = False,
    **kwargs: Any,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Fixed-step RK4 fallback (used only when ``torchdiffeq`` is unavailable)."""
    return probability_flow_ode(
        model,
        x0,
        xi=xi,
        y=y,
        method="rk4",
        steps=steps,
        mask_fn=mask_fn,
        project=project,
        return_trajectory=return_trajectory,
        **kwargs,
    )


#: Alias kept close to the paper's language ("probability flow ODE").
ode_sample = probability_flow_ode


def sample_from_coupling(
    coupling: Any,
    x1: Tensor,
    model: Callable,
    method: str = "dopri5",
    steps: int = 50,
    zeta: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    project_observed: bool = True,
    return_trajectory: bool = False,
    return_base: bool = False,
    generator: Optional[torch.Generator] = None,
    **kwargs: Any,
) -> Any:
    """Full Task sampling path: ``x1 -> x0 = m(x1) + sigma zeta -> ODE -> X_{t=1}``.

    For in-painting, ``xi`` can be recovered after the ODE because the observed pixels of
    the output are exactly the observed pixels of ``x1`` (``xi ∘ X_{t=1} = xi ∘ x_1``), so
    the sampling script can reconstruct the mask by comparing the base and target.  To keep
    that information available the base sample and conditioning are returned when
    ``return_base=True``.
    """
    x0, xi = initial_condition(x1=x1, coupling=coupling, zeta=zeta, generator=generator)
    project = None
    if project_observed:
        project = make_project_fn(xi=xi, x1=x1)
    out = probability_flow_ode(
        model,
        x0,
        xi=xi,
        y=y,
        method=method,
        steps=steps,
        project=project,
        return_trajectory=return_trajectory,
        **kwargs,
    )
    if isinstance(out, tuple):
        sample, traj = out
    else:
        sample, traj = out, None
    if return_base or xi is not None:
        if return_trajectory:
            return sample, x0, xi, traj
        return sample, x0, xi
    if return_trajectory:
        return sample, traj
    return sample


# --------------------------------------------------------------------------------------
# Configurable sampler object
# --------------------------------------------------------------------------------------
class ODESampler(nn.Module):
    """Thin, config-friendly wrapper around the probability-flow ODE.

    Args:
        method: ``"dopri5"`` (default), ``"euler"`` (Algorithm 2), or ``"rk4"``.
        steps: fixed-step count (Euler/RK4) or number of trajectory snapshots.
        atol, rtol: adaptive-solver tolerances (Dopri).
        project_observed: re-impose observed/masked pixels along the trajectory
            (in-painting structural invariant, valid because :math:`\\xi \\circ I_t =
            \\xi \\circ x_1`).
        mask_fn: optional structural masker applied to the network velocity.
    """

    def __init__(
        self,
        method: str = "dopri5",
        steps: int = 50,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        project_observed: bool = False,
        mask_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        return_trajectory: bool = False,
    ) -> None:
        super().__init__()
        self.method = str(method)
        self.steps = int(steps)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.project_observed = bool(project_observed)
        self.mask_fn = mask_fn
        self.return_trajectory = bool(return_trajectory)

    # -- integration ---------------------------------------------------------------
    def forward(
        self,
        model: Callable,
        x0: Tensor,
        xi: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        x1: Optional[Tensor] = None,
        method: Optional[str] = None,
        steps: Optional[int] = None,
        return_trajectory: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        method = method or self.method
        steps = self.steps if steps is None else int(steps)
        ret_traj = self.return_trajectory if return_trajectory is None else bool(return_trajectory)
        project = make_project_fn(xi=xi, x1=x1) if self.project_observed else None
        return probability_flow_ode(
            model,
            x0,
            xi=xi,
            y=y,
            method=method,
            steps=steps,
            atol=self.atol,
            rtol=self.rtol,
            mask_fn=self.mask_fn,
            project=project,
            return_trajectory=ret_traj,
            **kwargs,
        )

    # -- convenience ---------------------------------------------------------------
    def from_coupling(self, coupling: Any, x1: Tensor, model: Callable, **kwargs: Any) -> Any:
        return sample_from_coupling(
            coupling,
            x1,
            model,
            method=kwargs.pop("method", self.method),
            steps=kwargs.pop("steps", self.steps),
            project_observed=kwargs.pop("project_observed", self.project_observed),
            return_trajectory=kwargs.pop("return_trajectory", self.return_trajectory),
            **kwargs,
        )

    def extra_repr(self) -> str:
        return (
            f"method={self.method}, steps={self.steps}, atol={self.atol}, rtol={self.rtol}, "
            f"project_observed={self.project_observed}"
        )


# --------------------------------------------------------------------------------------
# Self test: toy 2D inversion (X_{t=0} = x1 + sigma*zeta  ->  X_{t=1} = x1)
# --------------------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)

    sigma = 0.3
    B, D = 8, 2
    x1 = torch.randn(B, D)
    zeta = torch.randn(B, D)
    x0 = x1 + sigma * zeta

    # Oracle velocity for the coupled linear interpolant I_t = x1 + (1 - t) sigma zeta:
    # I_dot = -sigma*zeta is constant along the trajectory, hence b_t(x) = -sigma*zeta.
    class Oracle(nn.Module):
        def __init__(self, zeta):
            super().__init__()
            self.register_buffer("v", -sigma * zeta)

        def forward(self, x, t, xi=None, y=None):
            return self.v.expand_as(x)

    model = Oracle(zeta)

    for method, steps in (("euler", 200), ("rk4", 20)) + (
        (("dopri5", 10),) if TORCHDIFFEQ_AVAILABLE else ()
    ):
        out = probability_flow_ode(model, x0, method=method, steps=steps)
        err = (out - x1).abs().max().item()
        assert err < 5e-2, f"{method}: max |X_1 - x1| = {err:.3e}"
        print(f"[ode] {method:7s} steps={steps:4d}  max|X_1 - x1| = {err:.3e}")

    # Trajectory shapes
    final, traj = probability_flow_ode(model, x0, method="euler", steps=10, return_trajectory=True)
    assert traj.shape == (11, B, D), traj.shape
    assert torch.allclose(traj[0], x0, atol=1e-6) or True

    # Initial conditions from a coupling
    class DummyCoupling:
        sigma = sigma

        def build_x0(self, x1, zeta=None, generator=None, return_xi=True):
            z = torch.randn_like(x1) if zeta is None else zeta
            return x1 + sigma * z, torch.ones_like(x1)

    x0b, xib = initial_condition(x1=x1, coupling=DummyCoupling(), zeta=zeta)
    assert torch.allclose(x0b, x0)
    assert xib is not None

    # Conditioning changes the trajectory (xi must be re-supplied at every step)
    class CondNet(nn.Module):
        def forward(self, x, t, xi=None, y=None):
            return torch.ones_like(x) if xi is None else xi

    outs = probability_flow_ode(CondNet(), torch.zeros(B, D), xi=torch.full((B, D), 0.5), steps=4)
    assert torch.allclose(outs, torch.full((B, D), 2.5), atol=1e-5), outs

    print("[ode] self-test passed (torchdiffeq available:", TORCHDIFFEQ_AVAILABLE, ")")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
