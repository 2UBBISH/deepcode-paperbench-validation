"""Stochastic interpolants with data-dependent couplings.

This module implements Definition 3.1 (and its conditional generalisation,
Definition A.1) of

    Albergo, Goldstein, Boffi, Ranganath, Vanden-Eijnden,
    "Stochastic Interpolants with Data-Dependent Couplings", ICML 2024.

The stochastic interpolant is

    I_t = alpha_t x_0 + beta_t x_1 + gamma_t z,      t in [0, 1]      (eq. 1)

where (x_0, x_1) ~ rho(x_0, x_1) is a *coupled* pair of base/target samples
and z ~ N(0, Id) is independent of (x_0, x_1).  The coefficients satisfy

    alpha_0 = beta_1 = 1,  alpha_1 = beta_0 = gamma_0 = gamma_1 = 0,
    alpha_t^2 + beta_t^2 + gamma_t^2 > 0   for all t in [0, 1].

Because of these boundary conditions the marginal law of I_t interpolates
between the coupled base density rho_0 at t = 0 and the target density
rho_1 at t = 1, and both the velocity b_t(x) = E[I_dot_t | I_t = x] and the
score (through g_t(x) = E[z | I_t = x]) are minimisers of the quadratic
objectives L_b and L_g in Theorem 3.1.

Notes on conventions
--------------------
* The default schedule is the "simple instance" explicitly given in the
  paper: alpha_t = 1 - t, beta_t = t, gamma_t = sqrt(2 t (1 - t)).
* Section 4.1 of the paper writes "we set alpha_t = t and beta_t = 1 - t".
  That pair is the time-reversed labelling of the very same interpolant
  (with the roles of x_0 and x_1 exchanged), and it contradicts both the
  boundary conditions of Definition 3.1 and the sampling pseudo-code
  (Algorithm 2), which initialises X_0 = m(x_1) + sigma zeta and integrates
  forward to the clean sample.  We therefore follow Definition 3.1 and
  Algorithm 2 and use alpha_t = 1 - t, beta_t = t for all experiments.
  ``LinearInterpolant(reverse=True)`` reproduces the literal statement of
  Section 4.1 for completeness.
* Couplings of the generic form x_0 = m(x_1) + sigma zeta (eq. 18) are
  handled through the *effective* noise coefficient

      gamma_tilde_t = alpha_t * sigma,

  i.e. the interpolant is re-written as

      I_t = alpha_t m(x_1) + beta_t x_1 + gamma_tilde_t zeta,

  so that the score identity (eq. 6 / eq. 28) reads
  grad log rho_t(x) = -gamma_tilde_t^{-1} g_t(x).  This is the statement
  made at the end of Section 3.2 ("the score ... is still available because
  of the factor of sigma zeta, so long as sigma is invertible").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor


class Coeffs(NamedTuple):
    """Values of the interpolant coefficients and their time derivatives."""

    alpha: Tensor
    beta: Tensor
    gamma: Tensor
    dalpha: Tensor
    dbeta: Tensor
    dgamma: Tensor


def _bc(coeff: Tensor, data: Tensor) -> Tensor:
    """Broadcast a coefficient of shape (B,) against data of shape (B, ...)."""
    if coeff.dim() == 1 and data.dim() > 1:
        return coeff.reshape(coeff.shape[0], *([1] * (data.dim() - 1)))
    return coeff


class InterpolantSchedule(torch.nn.Module):
    """Base class for the time schedules (alpha_t, beta_t, gamma_t)."""

    name: str = "schedule"

    def raw(self, t: Tensor) -> Coeffs:
        raise NotImplementedError

    def coefficients(self, t: Tensor) -> Coeffs:
        """Return the coefficients for a 1-D ``t`` of shape (B,).

        Parameters
        ----------
        t: Tensor of shape (B,) (or any shape broadcastable to the data).

        Returns
        -------
        Coeffs whose tensors all have shape (B,).  The helpers
        :meth:`interpolate` and :meth:`interpolate_velocity` broadcast them
        against vector- or image-shaped data automatically.
        """
        if t.dim() == 0:
            t = t[None]
        return self.raw(t)

    # -- convenience ------------------------------------------------------
    def interpolate(self, t: Tensor, x0: Tensor, x1: Tensor, z: Tensor | None = None) -> Tensor:
        """I_t = alpha_t x_0 + beta_t x_1 + gamma_t z (eq. 1)."""
        c = self.coefficients(t)
        out = _bc(c.alpha, x0) * x0 + _bc(c.beta, x0) * x1
        if z is not None:
            out = out + _bc(c.gamma, x0) * z
        return out

    def interpolate_velocity(
        self, t: Tensor, x0: Tensor, x1: Tensor, z: Tensor | None = None
    ) -> Tensor:
        """I_dot_t = alpha_dot_t x_0 + beta_dot_t x_1 + gamma_dot_t z."""
        c = self.coefficients(t)
        out = _bc(c.dalpha, x0) * x0 + _bc(c.dbeta, x0) * x1
        if z is not None:
            out = out + _bc(c.dgamma, x0) * z
        return out

    def equivalent_velocity(
        self, t: Tensor, m1: Tensor, x1: Tensor, zeta: Tensor, sigma: Tensor | float
    ) -> Tensor:
        """I_dot_t for the coupling x_0 = m(x_1) + sigma zeta (eq. 18/19/20).

        Note that the *noise* contribution to I_dot_t is
        d(alpha_t)/dt * sigma zeta because gamma_t = 0 in this construction.
        """
        c = self.coefficients(t)
        return _bc(c.dalpha, x1) * (m1 + sigma * zeta) + _bc(c.dbeta, x1) * x1

    def effective_gamma(self, t: Tensor, sigma: Tensor | float) -> Tensor:
        """gamma_tilde_t = alpha_t * sigma, the noise scale of the coupling."""
        c = self.coefficients(t)
        return c.alpha * sigma

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r})"


class LinearInterpolant(InterpolantSchedule):
    """alpha_t = 1 - t, beta_t = t, gamma_t = sqrt(2 t (1 - t)).

    This is the default instantiation of Definition 3.1 in the paper.  With
    ``with_variance=False`` the coefficient gamma_t is set to zero, which is
    the choice used in all of the coupling experiments (Section 3.2/4): the
    randomness of the base is carried by the coupling itself, so no extra
    z ~ N(0, Id) is needed (and the loss becomes a plain L2 regression
    against I_dot_t = x_1 - x_0).
    """

    def __init__(self, with_variance: bool = True, reverse: bool = False):
        super().__init__()
        self.with_variance = bool(with_variance)
        self.reverse = bool(reverse)
        if reverse:
            self.name = "linear_reversed"
        else:
            self.name = "linear" if with_variance else "linear_zero_gamma"

    def raw(self, t: Tensor) -> Coeffs:
        one = torch.ones_like(t)
        if not self.reverse:
            alpha = one - t
            beta = t
            dalpha = -one
            dbeta = one
        else:  # literal statement of Section 4.1 (time-reversed labelling)
            alpha = t
            beta = one - t
            dalpha = one
            dbeta = -one
        if self.with_variance:
            gamma = torch.sqrt(torch.clamp(2.0 * t * (one - t), min=0.0))
            dgamma = (one - 2.0 * t) / torch.clamp(gamma, min=1e-8)
        else:
            gamma = torch.zeros_like(t)
            dgamma = torch.zeros_like(t)
        return Coeffs(alpha, beta, gamma, dalpha, dbeta, dgamma)


class VPInterpolant(InterpolantSchedule):
    """Variance-preserving (diffusion-like) interpolant.

    alpha_t = cos(pi t / 2), beta_t = sin(pi t / 2), gamma_t = 0.

    Provided for completeness: it satisfies the boundary conditions of
    Definition 3.1 and can be used with any of the couplings implemented in
    :mod:`si_couplings.couplings`.
    """

    name = "vp"

    def raw(self, t: Tensor) -> Coeffs:
        half_pi = math.pi / 2.0
        alpha = torch.cos(half_pi * t)
        beta = torch.sin(half_pi * t)
        dalpha = -half_pi * torch.sin(half_pi * t)
        dbeta = half_pi * torch.cos(half_pi * t)
        zeros = torch.zeros_like(t)
        return Coeffs(alpha, beta, zeros, dalpha, dbeta, zeros)


def build_interpolant(name: str = "linear", **kwargs) -> InterpolantSchedule:
    """Factory for the schedules used in the paper.

    ``name`` is one of ``"linear"`` (default, gamma_t = sqrt(2t(1-t))),
    ``"linear_zero_gamma"`` (gamma_t = 0, used with couplings),
    ``"linear_reversed"`` (the literal schedule of Section 4.1) and ``"vp"``.
    """
    name = name.lower()
    if name in ("linear", "linear_zero_gamma", "linear_reversed"):
        return LinearInterpolant(
            with_variance=(name != "linear_zero_gamma"), reverse=(name == "linear_reversed")
        )
    if name == "vp":
        return VPInterpolant()
    raise ValueError(f"unknown interpolant schedule {name!r}")


# ---------------------------------------------------------------------------
# Analytic reference quantities (used by the unit tests and the toy studies)
# ---------------------------------------------------------------------------
@dataclass
class GaussianCoupling:
    """A jointly Gaussian coupling (x_0, x_1) ~ N(mu, Sigma).

    For such a coupling every quantity appearing in Theorem 3.1 is available
    in closed form, which makes it a useful reference for the unit tests:

        I_t        = alpha_t x_0 + beta_t x_1 + gamma_t z
        E[I_t]     = alpha_t mu_0 + beta_t mu_1
        Cov(I_t)   = alpha^2 C_00 + beta^2 C_11
                     + alpha beta (C_01 + C_01^T) + gamma^2 Id
        b_t(x)     = E[I_dot_t | I_t = x]
                   = E[I_dot_t] + Cov(I_dot_t, I_t) Cov(I_t)^{-1} (x - E[I_t])
        g_t(x)     = E[z | I_t = x]
                   = gamma_t Cov(I_t)^{-1} (x - E[I_t])
    """

    mu_0: Tensor
    mu_1: Tensor
    C_00: Tensor
    C_11: Tensor
    C_01: Tensor

    # -- geometry of the coupling ----------------------------------------
    @property
    def dim(self) -> int:
        return int(self.mu_0.shape[-1])

    def mean(self, alpha: Tensor, beta: Tensor) -> Tensor:
        return alpha * self.mu_0 + beta * self.mu_1

    def mean_velocity(self, dalpha: Tensor, dbeta: Tensor) -> Tensor:
        return dalpha * self.mu_0 + dbeta * self.mu_1

    def cov(self, alpha: Tensor, beta: Tensor, gamma_sq: Tensor) -> Tensor:
        eye = torch.eye(self.dim, dtype=self.C_00.dtype, device=self.C_00.device)
        return (
            alpha**2 * self.C_00
            + beta**2 * self.C_11
            + alpha * beta * (self.C_01 + self.C_01.transpose(-1, -2))
            + gamma_sq * eye
        )

    def cross_cov(
        self,
        alpha: float,
        beta: float,
        dalpha: float,
        dbeta: float,
        gamma: float = 0.0,
        dgamma: float = 0.0,
    ) -> Tensor:
        """Cov(I_dot_t, I_t) = E[(I_dot - E I_dot)(I_t - E I_t)^T].

        The contribution of the independent interpolant noise z is
        dgamma * gamma * Id (Cov(I_dot, I_t) = ... + gamma_dot gamma Id).
        """
        eye = torch.eye(self.dim, dtype=self.C_00.dtype, device=self.C_00.device)
        return (
            dalpha * alpha * self.C_00
            + dbeta * beta * self.C_11
            + dbeta * alpha * self.C_01
            + dalpha * beta * self.C_01.transpose(-1, -2)
            + dgamma * gamma * eye
        )

    def optimal_velocity(
        self,
        x: Tensor,
        alpha: float,
        beta: float,
        dalpha: float,
        dbeta: float,
        gamma_sq: float = 0.0,
        gamma: float = 0.0,
        dgamma: float = 0.0,
    ) -> Tensor:
        """b_t(x) = E[I_dot_t | I_t = x] for a jointly Gaussian coupling."""
        mean = self.mean(alpha, beta)
        cov = self.cov(alpha, beta, torch.as_tensor(gamma_sq, dtype=x.dtype, device=x.device))
        gain = self.cross_cov(alpha, beta, dalpha, dbeta, gamma, dgamma) @ torch.linalg.pinv(cov)
        return self.mean_velocity(dalpha, dbeta) + (x - mean) @ gain.T

    def optimal_score_scale(
        self,
        x: Tensor,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> Tensor:
        """-gamma_t^{-1} g_t(x) = grad log rho_t(x) for the Gaussian coupling."""
        mean = self.mean(alpha, beta)
        cov = self.cov(alpha, beta, torch.as_tensor(gamma**2, dtype=x.dtype, device=x.device))
        return -(x - mean) @ torch.linalg.pinv(cov).T

    # -- convenience wrappers that take an interpolant schedule -----------
    def velocity_from_schedule(self, x: Tensor, schedule: "InterpolantSchedule", t: float) -> Tensor:
        tb = torch.full((x.shape[0],), float(t))
        c = schedule.coefficients(tb)
        return self.optimal_velocity(
            x,
            float(c.alpha[0]),
            float(c.beta[0]),
            float(c.dalpha[0]),
            float(c.dbeta[0]),
            float(c.gamma[0]) ** 2,
            float(c.gamma[0]),
            float(c.dgamma[0]),
        )

    def log_score_from_schedule(self, x: Tensor, schedule: "InterpolantSchedule", t: float) -> Tensor:
        """grad log rho_t(x) = -Sigma_t^{-1}(x - mu_t) for a Gaussian coupling."""
        tb = torch.full((x.shape[0],), float(t))
        c = schedule.coefficients(tb)
        mean = self.mean(float(c.alpha[0]), float(c.beta[0]))
        cov = self.cov(float(c.alpha[0]), float(c.beta[0]), c.gamma[0] ** 2)
        return -(x - mean) @ torch.linalg.pinv(cov).T


__all__ = [
    "Coeffs",
    "InterpolantSchedule",
    "LinearInterpolant",
    "VPInterpolant",
    "GaussianCoupling",
    "build_interpolant",
]
