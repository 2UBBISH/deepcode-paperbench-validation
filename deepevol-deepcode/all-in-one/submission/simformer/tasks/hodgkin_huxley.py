"""Hodgkin-Huxley task (Simformer, Sec. 4.4 / Appendix A2.2).

Implements the Hodgkin-Huxley (HH) neuron simulator used for the
"inference with observation intervals" experiment of the Simformer paper.

Paper specification (Appendix A2.2, "Hodgkin-Huxley Model")
----------------------------------------------------------
* Implementation guidelines follow Pospischil et al. (2008).
* Initial membrane voltage ``V_0 = -65.0`` mV.
* Simulations run for 200 ms with an input current of 4 mA (= 4 uA/cm^2 with
  ``C_m`` measured in uF/cm^2) applied between 50 ms and 150 ms.
* Rate functions (with ``V_0`` the reference/resting potential)::

      alpha_m(V) = 0.32 * efun(-0.25 * (V - V_0 - 13.0)) / 0.25
      beta_m(V)  = 0.28 * efun( 0.2  * (V - V_0 - 40.0)) / 0.2
      alpha_h(V) = 0.128 * exp(-(V - V_0 - 17.0) / 18.0)
      beta_h(V)  = 4.0 / (1.0 + exp(-(V - V_0 - 40.0) / 5.0))
      alpha_n(V) = 0.032 * efun(-0.2 * (V - V_0 - 15.0)) / 0.2
      beta_n(V)  = 0.5 * exp(-(V - V_0 - 10.0) / 40.0)

  where ``efun(x) = 1 - x/2`` for ``x < 1e-4`` and ``x / (exp(x) - 1)``
  otherwise.
* State equations (paper eq. 10)::

      dV/dt = (I_inj(t) - g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K)
               - g_L (V - E_L)) / C_m + 0.05 dW_t
      dm/dt = alpha_m(V) (1 - m) - beta_m(V) m
      dh/dt = alpha_h(V) (1 - h) - beta_h(V) h
      dn/dt = alpha_n(V) (1 - n) - beta_n(V) n
      dH/dt = g_Na m^3 h (V - E_Na)

* The simulator has 7 parameters.
* ``H`` accumulates the *sodium charge*; following Deistler et al. (2022b) it
  is converted to metabolic cost in ``uJ/s`` (:func:`convert_charge_to_energy`).
* Observational data are summary statistics of the voltage trace, following
  Goncalves et al. (2020); the metabolic cost (energy) is recorded as an
  additional statistic (Sec. 4.4).

Summary statistics (Goncalves et al. 2020 style, 7 features)
-----------------------------------------------------------
``[spike_count, resting_mean, resting_std, spiking_mean,
spiking_2nd_moment, spiking_3rd_moment, spiking_4th_moment]``

Where ``resting`` refers to the pre-stimulus window and ``spiking`` to the
stimulus window; the moments are central moments about the spiking mean.

Where the paper is silent
-------------------------
* The parameter prior is not printed in the paper. We use the uniform prior of
  the Goncalves et al. (2020) HH benchmark (see :data:`DEFAULT_PRIOR_RANGES`);
  it is exposed through :class:`HodgkinHuxleyConfig` and can be overridden.
* The likelihood of the summary statistics is not available in closed form;
  for ground-truth MCMC we use a diagonal Gaussian model on the statistics
  centred at the (near deterministic) simulated statistics, with per-statistic
  standard deviations estimated from repeated stochastic simulations
  (``stats_sigma=None`` -> estimated once and cached) or supplied directly.
* The paper's printed "4 mA" input current is interpreted as 4 (current units
  per unit capacitance, i.e. uA/cm^2 with C_m in uF/cm^2); using 4 mA/cm^2
  would produce absurd (4000 mV/ms) trans-membrane currents.
* ``efun``'s ``1 - x/2`` branch is used for ``x < 1e-4`` exactly as printed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - registry base class (optional)
    from . import TaskBase
except Exception:  # pragma: no cover - standalone fallback
    try:
        from simformer.tasks import TaskBase  # type: ignore
    except Exception:
        class TaskBase:  # type: ignore
            """Minimal stand-in used when the task registry is unavailable."""

            name = "task"
            n_parameters = 0
            n_data = 0
            parameter_names: Tuple[str, ...] = ()
            data_names: Tuple[str, ...] = ()

            def to_joint(self, theta, x):
                theta = np.atleast_2d(np.asarray(theta, dtype=float))
                x = np.atleast_1d(np.asarray(x, dtype=float))
                if x.ndim == 1:
                    x = x[None]
                return np.concatenate([theta, x], axis=-1)

            def split_joint(self, joint):
                joint = np.atleast_2d(np.asarray(joint, dtype=float))
                return joint[:, : self.n_parameters], joint[:, self.n_parameters :]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DURATION = 200.0            # ms
DEFAULT_DT = 0.05                   # ms (RK4 / Euler-Maruyama step)
DEFAULT_V0 = -65.0                  # mV
DEFAULT_I_INJ = 4.0                 # uA/cm^2 (paper prints "4 mA")
DEFAULT_I_INJ_START = 50.0          # ms
DEFAULT_I_INJ_END = 150.0           # ms
DEFAULT_NOISE_SIGMA = 0.05          # mV / sqrt(ms)   (paper: 0.05 dW_t)

DEFAULT_PARAMETER_NAMES: Tuple[str, ...] = (
    "g_Na",
    "g_K",
    "g_L",
    "E_Na",
    "E_K",
    "E_L",
    "C_m",
)

#: Uniform prior ranges (Goncalves et al., 2020 HH benchmark).
DEFAULT_PRIOR_RANGES: Dict[str, Tuple[float, float]] = {
    "g_Na": (10.0, 100.0),   # mS/cm^2
    "g_K": (1.0, 20.0),      # mS/cm^2
    "g_L": (0.01, 0.1),      # mS/cm^2
    "E_Na": (30.0, 70.0),    # mV
    "E_K": (-100.0, -70.0),  # mV
    "E_L": (-90.0, -50.0),   # mV
    "C_m": (0.5, 2.5),       # uF/cm^2
}

DEFAULT_STAT_NAMES: Tuple[str, ...] = (
    "spike_count",
    "resting_mean",
    "resting_std",
    "spiking_mean",
    "spiking_2nd_moment",
    "spiking_3rd_moment",
    "spiking_4th_moment",
)
ENERGY_STAT_NAME = "energy"

DEFAULT_SPIKE_THRESHOLD = 0.0       # mV
DEFAULT_REFRACTORY = 2.0            # ms (spike-count dead time)
DEFAULT_RESTING_WINDOW = (10.0, 50.0)    # ms (skip initial transient)
DEFAULT_SPIKING_WINDOW = (50.0, 150.0)   # ms (stimulus window)

DEFAULT_N_GLOBAL_SIMULATIONS = 5000

# --- energy conversion (Deistler et al., 2022b) ---------------------------
FARADAY_CONSTANT = 96485.33212      # C / mol
ATP_ENERGY_J_PER_MOL = 50_000.0     # free energy of ATP hydrolysis (~50 kJ/mol)
NA_IONS_PER_ATP = 3.0               # Na/K-ATPase stoichiometry (3 Na+ : 1 ATP)
J_PER_COULOMB = ATP_ENERGY_J_PER_MOL / (NA_IONS_PER_ATP * FARADAY_CONSTANT)
DEFAULT_ENERGY_AREA_CM2 = 1.0       # membrane area used for the reported cost

ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# Rate functions (paper eq. 9)
# ---------------------------------------------------------------------------

def efun(x: ArrayLike) -> np.ndarray:
    """``efun(x) = 1 - x/2`` if ``x < 1e-4`` else ``x / (exp(x) - 1)``."""
    x = np.asarray(x, dtype=float)
    tiny = x < 1e-4
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denom = np.expm1(np.where(tiny, 1.0, x))
        safe = np.where(tiny, 1.0, x / denom)
    return np.where(tiny, 1.0 - 0.5 * x, safe)


def alpha_m(V: ArrayLike, v0: float = DEFAULT_V0) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    return 0.32 * efun(-0.25 * (V - v0 - 13.0)) / 0.25


def beta_m(V: ArrayLike, v0: float = DEFAULT_V0) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    return 0.28 * efun(0.2 * (V - v0 - 40.0)) / 0.2


def alpha_h(V: ArrayLike, v0: float = DEFAULT_V0) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    return 0.128 * np.exp(-(V - v0 - 17.0) / 18.0)


def beta_h(V: ArrayLike, v0: float = DEFAULT_V0) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    with np.errstate(over="ignore"):
        return 4.0 / (1.0 + np.exp(-(V - v0 - 40.0) / 5.0))


def alpha_n(V: ArrayLike, v0: float = DEFAULT_V0) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    return 0.032 * efun(-0.2 * (V - v0 - 15.0)) / 0.2


def beta_n(V: ArrayLike, v0: float = DEFAULT_V0) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    return 0.5 * np.exp(-(V - v0 - 10.0) / 40.0)


def steady_state_gating(V: float, v0: float = DEFAULT_V0) -> np.ndarray:
    """Steady-state gating values ``(m_inf, h_inf, n_inf)`` at voltage ``V``."""
    V = float(V)
    am, bm = float(alpha_m(V, v0)), float(beta_m(V, v0))
    ah, bh = float(alpha_h(V, v0)), float(beta_h(V, v0))
    an, bn = float(alpha_n(V, v0)), float(beta_n(V, v0))
    return np.array(
        [
            am / (am + bm),
            ah / (ah + bh),
            an / (an + bn),
        ],
        dtype=float,
    )


def gating_derivatives(
    V: ArrayLike, m: ArrayLike, h: ArrayLike, n: ArrayLike, v0: float = DEFAULT_V0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(dm/dt, dh/dt, dn/dt)`` from the gating equations (paper eq. 10)."""
    V, m, h, n = (np.asarray(a, dtype=float) for a in (V, m, h, n))
    am, bm = alpha_m(V, v0), beta_m(V, v0)
    ah, bh = alpha_h(V, v0), beta_h(V, v0)
    an, bn = alpha_n(V, v0), beta_n(V, v0)
    return (
        am * (1.0 - m) - bm * m,
        ah * (1.0 - h) - bh * h,
        an * (1.0 - n) - bn * n,
    )


def stimulus_current(
    t: ArrayLike,
    *,
    amplitude: float = DEFAULT_I_INJ,
    start: float = DEFAULT_I_INJ_START,
    end: float = DEFAULT_I_INJ_END,
) -> np.ndarray:
    """Square-pulse injected current (paper: 4 applied in [50 ms, 150 ms])."""
    t = np.asarray(t, dtype=float)
    return np.where((t >= start) & (t <= end), float(amplitude), 0.0)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def hodgkin_huxley_rhs(
    state: ArrayLike,
    t: float,
    theta: ArrayLike,
    *,
    v0: float = DEFAULT_V0,
    i_inj: float = DEFAULT_I_INJ,
    i_inj_start: float = DEFAULT_I_INJ_START,
    i_inj_end: float = DEFAULT_I_INJ_END,
) -> np.ndarray:
    """Right-hand side of the HH ODE system ``[V, m, h, n, H]`` (paper eq. 10)."""
    state = np.asarray(state, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    V, m, h, n, _H = state
    g_na, g_k, g_l, e_na, e_k, e_l, c_m = theta

    i_na = g_na * (m ** 3) * h * (V - e_na)
    i_k = g_k * (n ** 4) * (V - e_k)
    i_l = g_l * (V - e_l)

    dv = (
        float(stimulus_current(t, amplitude=i_inj, start=i_inj_start, end=i_inj_end))
        - i_na
        - i_k
        - i_l
    ) / c_m
    dm, dh, dn = gating_derivatives(V, m, h, n, v0)
    return np.array([dv, float(dm), float(dh), float(dn), float(i_na)], dtype=float)


def integrate_hodgkin_huxley(
    theta: ArrayLike,
    *,
    duration: float = DEFAULT_DURATION,
    dt: float = DEFAULT_DT,
    n_steps: Optional[int] = None,
    v0: float = DEFAULT_V0,
    i_inj: float = DEFAULT_I_INJ,
    i_inj_start: float = DEFAULT_I_INJ_START,
    i_inj_end: float = DEFAULT_I_INJ_END,
    initial_state: Optional[ArrayLike] = None,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    stochastic: bool = True,
    method: str = "auto",
    rng: Optional[np.random.Generator] = None,
    return_times: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Integrate the HH system, returning the ``(n_steps + 1, 5)`` trajectory.

    Columns are ``[V, m, h, n, H]`` where ``H`` is the accumulated sodium charge
    (nC/cm^2).  ``stochastic=True`` adds the paper's ``0.05 dW_t`` noise term to
    the voltage equation (Euler-Maruyama); otherwise an RK4 step is used.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != 7:
        raise ValueError(f"theta must have 7 entries, got {theta.size}")
    if n_steps is None:
        n_steps = int(round(float(duration) / float(dt)))
    n_steps = max(int(n_steps), 1)
    times = np.linspace(0.0, float(duration), n_steps + 1)

    if initial_state is None:
        gating = steady_state_gating(v0, v0)
        initial_state = np.array([v0, gating[0], gating[1], gating[2], 0.0])
    state = np.asarray(initial_state, dtype=float).reshape(-1).copy()
    if state.size != 5:
        raise ValueError("initial_state must have 5 entries [V, m, h, n, H]")

    if method == "auto":
        method = "euler" if (stochastic and noise_sigma > 0.0) else "rk4"

    traj = np.empty((n_steps + 1, 5), dtype=float)
    traj[0] = state
    sqrt_dt = math.sqrt(float(dt))
    for i in range(n_steps):
        t = times[i]
        if method == "rk4":
            k1 = hodgkin_huxley_rhs(state, t, theta, v0=v0, i_inj=i_inj,
                                    i_inj_start=i_inj_start, i_inj_end=i_inj_end)
            k2 = hodgkin_huxley_rhs(state + 0.5 * dt * k1, t + 0.5 * dt, theta, v0=v0,
                                    i_inj=i_inj, i_inj_start=i_inj_start,
                                    i_inj_end=i_inj_end)
            k3 = hodgkin_huxley_rhs(state + 0.5 * dt * k2, t + 0.5 * dt, theta, v0=v0,
                                    i_inj=i_inj, i_inj_start=i_inj_start,
                                    i_inj_end=i_inj_end)
            k4 = hodgkin_huxley_rhs(state + dt * k3, t + dt, theta, v0=v0,
                                    i_inj=i_inj, i_inj_start=i_inj_start,
                                    i_inj_end=i_inj_end)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:  # Euler-Maruyama (paper includes the noise term on dV/dt only)
            drift = hodgkin_huxley_rhs(state, t, theta, v0=v0, i_inj=i_inj,
                                       i_inj_start=i_inj_start, i_inj_end=i_inj_end)
            state = state + dt * drift
            if noise_sigma > 0.0:
                if rng is None:
                    rng = np.random.default_rng()
                state[0] += float(noise_sigma) * sqrt_dt * float(rng.standard_normal())
        # keep gating variables in [0, 1] for numerical robustness
        np.clip(state[1:4], 0.0, 1.0, out=state[1:4])
        traj[i + 1] = state
    return (traj, times) if return_times else traj


def integrate_trajectories(
    thetas: ArrayLike,
    *,
    duration: float = DEFAULT_DURATION,
    dt: float = DEFAULT_DT,
    n_steps: Optional[int] = None,
    stochastic: bool = True,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    rng: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the HH system for a batch of parameters.

    Returns ``(trajectories, times)`` with trajectories of shape
    ``(n_batch, n_steps + 1, 5)``.
    """
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    trajs: List[np.ndarray] = []
    times: np.ndarray = np.array([])
    for th in thetas:
        traj, times = integrate_hodgkin_huxley(
            th, duration=duration, dt=dt, n_steps=n_steps, stochastic=stochastic,
            noise_sigma=noise_sigma, rng=rng, **kwargs
        )
        trajs.append(traj)
    return np.stack(trajs, axis=0), times


# ---------------------------------------------------------------------------
# Summary statistics and energy (Sec. 4.4 / Goncalves et al. 2020)
# ---------------------------------------------------------------------------

def spike_times(
    voltage: ArrayLike,
    times: ArrayLike,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    refractory: float = DEFAULT_REFRACTORY,
) -> np.ndarray:
    """Upward threshold crossings, with a refractory dead-time."""
    voltage = np.asarray(voltage, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    if voltage.size < 2:
        return np.zeros(0, dtype=float)
    above = voltage >= float(threshold)
    crossings = np.nonzero((~above[:-1]) & above[1:])[0]
    if crossings.size == 0:
        return np.zeros(0, dtype=float)
    accepted: List[float] = []
    last = -np.inf
    for idx in crossings:
        t_cross = float(times[idx + 1])
        if t_cross - last > float(refractory):
            accepted.append(t_cross)
            last = t_cross
    return np.asarray(accepted, dtype=float)


def spike_count(
    voltage: ArrayLike,
    times: ArrayLike,
    *,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    refractory: float = DEFAULT_REFRACTORY,
) -> int:
    """Number of spikes (plateau crossings with refractory dead-time)."""
    return int(spike_times(voltage, times, threshold=threshold,
                           refractory=refractory).size)


def central_moments(x: ArrayLike, orders: Sequence[int] = (2, 3, 4)) -> np.ndarray:
    """Central moments of ``x`` (population moments, i.e. no Bessel correction)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.zeros(len(orders), dtype=float)
    mean = float(np.mean(x))
    return np.array([float(np.mean((x - mean) ** k)) for k in orders], dtype=float)


def sodium_charge(trajectory: ArrayLike, *, take_last: bool = True) -> np.ndarray:
    """Accumulated sodium charge ``H`` in nC/cm^2 (last column of the state)."""
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim == 1:
        traj = traj[None]
    return traj[..., -1] if take_last else traj[..., 4]


def convert_charge_to_energy(
    charge: ArrayLike,
    *,
    duration: float = DEFAULT_DURATION,
    area_cm2: float = DEFAULT_ENERGY_AREA_CM2,
    j_per_coulomb: float = J_PER_COULOMB,
    return_watts: bool = False,
) -> np.ndarray:
    """Convert sodium charge (nC/cm^2) into metabolic cost (uJ/s).

    ``charge`` is the accumulated sodium charge density (the HH state ``H``) in
    nC/cm^2, whose average current density is ``charge / duration`` in
    uA/cm^2 (Deistler et al., 2022b use the sodium charge for the energy
    estimate).  Moving one Na+ ion costs one third of an ATP molecule, so the
    energy per unit charge is ``ATP_ENERGY_J_PER_MOL / (3 * F)`` J/C.  With a
    membrane area ``area_cm2`` this yields an energy flux that we report in
    uJ/s (i.e. uW) following the paper.
    """
    charge = np.asarray(charge, dtype=float)
    mean_current_uA_cm2 = charge / max(float(duration), 1e-12)
    energy_uJ_per_s = float(j_per_coulomb) * mean_current_uA_cm2 * float(area_cm2)
    if return_watts:
        return energy_uJ_per_s * 1e-6
    return energy_uJ_per_s


def energy_from_trajectory(
    trajectory: ArrayLike,
    *,
    duration: float = DEFAULT_DURATION,
    area_cm2: float = DEFAULT_ENERGY_AREA_CM2,
    j_per_coulomb: float = J_PER_COULOMB,
) -> float:
    """Metabolic cost (uJ/s) implied by a simulated trajectory."""
    charge = float(np.asarray(sodium_charge(trajectory), dtype=float).reshape(-1)[-1])
    return float(
        convert_charge_to_energy(
            charge, duration=duration, area_cm2=area_cm2,
            j_per_coulomb=j_per_coulomb,
        )
    )


def summary_statistics(
    voltage: ArrayLike,
    times: ArrayLike,
    *,
    resting_window: Tuple[float, float] = DEFAULT_RESTING_WINDOW,
    spiking_window: Tuple[float, float] = DEFAULT_SPIKING_WINDOW,
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
    refractory: float = DEFAULT_REFRACTORY,
    energy: Optional[float] = None,
) -> np.ndarray:
    """Voltage summary statistics (7 features) plus optional energy (8th).

    ``[spike_count, resting_mean, resting_std, spiking_mean, m2, m3, m4]``
    with ``m_k`` the k-th central moment of the spiking-window voltage.
    """
    voltage = np.asarray(voltage, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    n = min(voltage.size, times.size)
    voltage, times = voltage[:n], times[:n]

    rest_lo, rest_hi = float(resting_window[0]), float(resting_window[1])
    spike_lo, spike_hi = float(spiking_window[0]), float(spiking_window[1])
    rest_mask = (times >= rest_lo) & (times < rest_hi)
    spike_mask = (times >= spike_lo) & (times <= spike_hi)

    rest = voltage[rest_mask] if np.any(rest_mask) else voltage
    spiking = voltage[spike_mask] if np.any(spike_mask) else voltage

    features = [
        float(spike_count(voltage, times, threshold=threshold, refractory=refractory)),
        float(np.mean(rest)),
        float(np.std(rest)),
        float(np.mean(spiking)),
    ]
    features.extend(central_moments(spiking, orders=(2, 3, 4)).tolist())
    if energy is not None:
        features.append(float(energy))
    return np.asarray(features, dtype=float)


def hodgkin_huxley_features(
    theta: ArrayLike,
    *,
    duration: float = DEFAULT_DURATION,
    dt: float = DEFAULT_DT,
    stochastic: bool = True,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    rng: Optional[np.random.Generator] = None,
    include_energy: bool = True,
    return_trajectory: bool = False,
    **kwargs: Any,
):
    """Simulate one parameter vector and return its summary statistics."""
    traj, times = integrate_hodgkin_huxley(
        theta, duration=duration, dt=dt, stochastic=stochastic,
        noise_sigma=noise_sigma, rng=rng, return_times=True, **kwargs
    )
    energy = (
        energy_from_trajectory(traj, duration=duration) if include_energy else None
    )
    feats = summary_statistics(traj[:, 0], times, energy=energy)
    if return_trajectory:
        return feats, traj, times
    return feats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HodgkinHuxleyConfig:
    """Configuration of the Hodgkin-Huxley task."""

    name: str = "hodgkin_huxley"
    duration: float = DEFAULT_DURATION
    dt: float = DEFAULT_DT
    v0: float = DEFAULT_V0
    i_inj: float = DEFAULT_I_INJ
    i_inj_start: float = DEFAULT_I_INJ_START
    i_inj_end: float = DEFAULT_I_INJ_END
    noise_sigma: float = DEFAULT_NOISE_SIGMA
    parameter_names: Tuple[str, ...] = DEFAULT_PARAMETER_NAMES
    prior_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    include_energy: bool = True
    spike_threshold: float = DEFAULT_SPIKE_THRESHOLD
    refractory: float = DEFAULT_REFRACTORY
    resting_window: Tuple[float, float] = DEFAULT_RESTING_WINDOW
    spiking_window: Tuple[float, float] = DEFAULT_SPIKING_WINDOW
    stats_sigma: Optional[Tuple[float, ...]] = None
    energy_area_cm2: float = DEFAULT_ENERGY_AREA_CM2
    j_per_coulomb: float = J_PER_COULOMB
    reference_steps: int = 5000
    reference_step_size: Optional[float] = None
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived ----------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    @property
    def n_stats(self) -> int:
        return len(DEFAULT_STAT_NAMES) + (1 if self.include_energy else 0)

    @property
    def n_data(self) -> int:
        return self.n_stats

    @property
    def data_names(self) -> Tuple[str, ...]:
        names = tuple(DEFAULT_STAT_NAMES)
        return names + ((ENERGY_STAT_NAME,) if self.include_energy else ())

    @property
    def energy_index(self) -> Optional[int]:
        return (len(DEFAULT_STAT_NAMES) if self.include_energy else None)

    def resolved_prior_ranges(self) -> Dict[str, Tuple[float, float]]:
        ranges = dict(DEFAULT_PRIOR_RANGES)
        if self.prior_ranges:
            ranges.update({str(k): tuple(v) for k, v in self.prior_ranges.items()})
        return ranges

    def prior_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        ranges = self.resolved_prior_ranges()
        low = np.array([ranges[n][0] for n in self.parameter_names], dtype=float)
        high = np.array([ranges[n][1] for n in self.parameter_names], dtype=float)
        return low, high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration": self.duration,
            "dt": self.dt,
            "v0": self.v0,
            "i_inj": self.i_inj,
            "i_inj_start": self.i_inj_start,
            "i_inj_end": self.i_inj_end,
            "noise_sigma": self.noise_sigma,
            "parameter_names": tuple(self.parameter_names),
            "prior_ranges": self.resolved_prior_ranges(),
            "include_energy": self.include_energy,
            "spike_threshold": self.spike_threshold,
            "refractory": self.refractory,
            "resting_window": tuple(self.resting_window),
            "spiking_window": tuple(self.spiking_window),
            "stats_sigma": self.stats_sigma,
            "energy_area_cm2": self.energy_area_cm2,
            "j_per_coulomb": self.j_per_coulomb,
            "reference_steps": self.reference_steps,
            "reference_step_size": self.reference_step_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(
        cls, cfg: Optional[Union[Dict[str, Any], "HodgkinHuxleyConfig"]] = None, **kwargs: Any
    ) -> "HodgkinHuxleyConfig":
        if isinstance(cfg, HodgkinHuxleyConfig):
            return _dc_replace(cfg, **kwargs) if kwargs else cfg
        data: Dict[str, Any] = {}
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        if cfg:
            for key, value in dict(cfg).items():
                if key in known:
                    data[key] = value
                else:
                    data.setdefault("extra", {})[key] = value
        for key, value in kwargs.items():
            if key in known:
                data[key] = value
            else:
                data.setdefault("extra", {})[key] = value
        if "parameter_names" in data and data["parameter_names"] is not None:
            data["parameter_names"] = tuple(data["parameter_names"])
        if "stats_sigma" in data and data["stats_sigma"] is not None:
            data["stats_sigma"] = tuple(float(s) for s in data["stats_sigma"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(0 if seed is None else int(seed))


def _as_2d_theta(theta: ArrayLike) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta[None]
    return theta


def _as_2d_x(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None]
    return x


def _diag_gaussian_logpdf(x: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Sum over the last axis of a diagonal Gaussian log-density."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    diff = (np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)) / sigma
    return -0.5 * np.sum(diff ** 2 + 2.0 * np.log(sigma) + math.log(2.0 * math.pi), axis=-1)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class HodgkinHuxleyTask(TaskBase):
    """Hodgkin-Huxley simulator task (Sec. 4.4, Appendix A2.2)."""

    name = "hodgkin_huxley"
    n_parameters = len(DEFAULT_PARAMETER_NAMES)
    parameter_names = DEFAULT_PARAMETER_NAMES
    data_names = tuple(DEFAULT_STAT_NAMES) + (ENERGY_STAT_NAME,)
    n_data = len(data_names)

    def __init__(
        self,
        config: Optional[Union[HodgkinHuxleyConfig, Dict[str, Any]]] = None,
        *,
        duration: Optional[float] = None,
        dt: Optional[float] = None,
        v0: Optional[float] = None,
        i_inj: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        include_energy: Optional[bool] = None,
        prior_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        stats_sigma: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        overrides: Dict[str, Any] = dict(kwargs)
        if duration is not None:
            overrides["duration"] = float(duration)
        if dt is not None:
            overrides["dt"] = float(dt)
        if v0 is not None:
            overrides["v0"] = float(v0)
        if i_inj is not None:
            overrides["i_inj"] = float(i_inj)
        if noise_sigma is not None:
            overrides["noise_sigma"] = float(noise_sigma)
        if include_energy is not None:
            overrides["include_energy"] = bool(include_energy)
        if prior_ranges is not None:
            overrides["prior_ranges"] = dict(prior_ranges)
        if stats_sigma is not None:
            overrides["stats_sigma"] = tuple(float(s) for s in stats_sigma)
        if name is not None:
            overrides["name"] = str(name)
        if seed is not None:
            overrides["seed"] = int(seed)
        self.config = HodgkinHuxleyConfig.from_dict(config, **overrides)
        self.name = self.config.name
        self.n_parameters = self.config.n_parameters
        self.parameter_names = tuple(self.config.parameter_names)
        self.data_names = self.config.data_names
        self.n_data = self.config.n_data
        self._stats_sigma: Optional[np.ndarray] = (
            None if self.config.stats_sigma is None
            else np.asarray(self.config.stats_sigma, dtype=float)
        )
        self._default_task: Optional[HodgkinHuxleyTask] = None

    # -- basic description ------------------------------------------------
    @property
    def n_stats(self) -> int:
        return self.config.n_stats

    @property
    def energy_index(self) -> Optional[int]:
        return self.config.energy_index

    def prior_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.config.prior_bounds()

    def feature_names(self) -> Tuple[str, ...]:
        return tuple(self.data_names)

    def to_dict(self) -> Dict[str, Any]:
        d = self.config.to_dict()
        d.update({"n_parameters": self.n_parameters, "n_data": self.n_data})
        return d

    # -- prior ------------------------------------------------------------
    def prior_sample(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = _rng(rng)
        low, high = self.prior_bounds()
        n_samples = int(max(n_samples, 1))
        return rng.uniform(low, high, size=(n_samples, low.size))

    sample_prior = prior_sample

    def log_prior(self, theta: ArrayLike) -> np.ndarray:
        theta = _as_2d_theta(theta)
        low, high = self.prior_bounds()
        inside = np.all((theta >= low - 1e-12) & (theta <= high + 1e-12), axis=-1)
        log_vol = float(np.sum(np.log(high - low)))
        return np.where(inside, -log_vol, -np.inf)

    # -- simulation -------------------------------------------------------
    def simulate(
        self,
        theta: ArrayLike,
        rng: Optional[np.random.Generator] = None,
        *,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        return_trajectory: bool = False,
        include_energy: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Simulate summary statistics of the voltage (and energy) for ``theta``.

        ``theta`` may be a single parameter vector (shape ``(7,)``) or a batch
        (shape ``(n, 7)``).  Returns an ``(n, n_data)`` array of statistics, or
        ``(features, trajectory, times)`` for the single-parameter case when
        ``return_trajectory`` is set.
        """
        rng = _rng(rng, seed)
        theta_arr = _as_2d_theta(theta)
        if n_samples is not None and theta_arr.shape[0] == 1 and int(n_samples) > 1:
            theta_arr = np.repeat(theta_arr, int(n_samples), axis=0)
        include = self.config.include_energy if include_energy is None else bool(include_energy)

        stochastic = bool(add_noise) and float(self.config.noise_sigma) > 0.0
        out: List[np.ndarray] = []
        last = None
        for th in theta_arr:
            feats, traj, times = hodgkin_huxley_features(
                th,
                duration=self.config.duration,
                dt=self.config.dt,
                stochastic=stochastic,
                noise_sigma=self.config.noise_sigma,
                rng=rng,
                include_energy=include,
                return_trajectory=True,
                v0=self.config.v0,
                i_inj=self.config.i_inj,
                i_inj_start=self.config.i_inj_start,
                i_inj_end=self.config.i_inj_end,
                spike_threshold=self.config.spike_threshold,
                refractory=self.config.refractory,
                resting_window=self.config.resting_window,
                spiking_window=self.config.spiking_window,
            )
            out.append(feats)
            last = (feats, traj, times)
        features = np.stack(out, axis=0)
        if return_trajectory and theta_arr.shape[0] == 1 and last is not None:
            return last[0], last[1], last[2]
        return features

    simulator = simulate

    def __call__(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None):
        rng = _rng(rng)
        theta = self.prior_sample(int(n_samples), rng)
        x = self.simulate(theta, rng)
        return theta, x

    def trajectory(
        self,
        theta: ArrayLike,
        *,
        add_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(trajectory, times)`` for one parameter vector."""
        rng = _rng(rng, seed)
        traj, times = integrate_hodgkin_huxley(
            np.asarray(theta, dtype=float).reshape(-1),
            duration=self.config.duration,
            dt=self.config.dt,
            stochastic=bool(add_noise),
            noise_sigma=self.config.noise_sigma,
            rng=rng,
            v0=self.config.v0,
            i_inj=self.config.i_inj,
            i_inj_start=self.config.i_inj_start,
            i_inj_end=self.config.i_inj_end,
            return_times=True,
        )
        return traj, times

    def voltage_trace(self, theta: ArrayLike, **kwargs: Any) -> np.ndarray:
        traj, _ = self.trajectory(theta, **kwargs)
        return traj[:, 0]

    def data_mean(self, theta: ArrayLike, *, include_energy: Optional[bool] = None) -> np.ndarray:
        """Deterministic (noise-free) summary statistics for ``theta``."""
        include = self.config.include_energy if include_energy is None else bool(include_energy)
        theta_arr = _as_2d_theta(theta)
        rows = [
            hodgkin_huxley_features(
                th,
                duration=self.config.duration,
                dt=self.config.dt,
                stochastic=False,
                noise_sigma=0.0,
                rng=None,
                include_energy=include,
                v0=self.config.v0,
                i_inj=self.config.i_inj,
                i_inj_start=self.config.i_inj_start,
                i_inj_end=self.config.i_inj_end,
                spike_threshold=self.config.spike_threshold,
                refractory=self.config.refractory,
                resting_window=self.config.resting_window,
                spiking_window=self.config.spiking_window,
            )
            for th in theta_arr
        ]
        return np.stack(rows, axis=0)

    # -- statistics noise / likelihood ------------------------------------
    def estimate_stats_sigma(
        self,
        *,
        n_thetas: int = 16,
        n_repeats: int = 2,
        seed: Optional[int] = None,
        floor: float = 1e-3,
    ) -> np.ndarray:
        """Estimate per-statistic noise std from repeated stochastic simulations."""
        rng = np.random.default_rng(self.config.seed if seed is None else int(seed))
        thetas = self.prior_sample(max(int(n_thetas), 2), rng)
        samples = np.stack(
            [self.simulate(thetas, rng) for _ in range(max(int(n_repeats), 1))],
            axis=0,
        )  # (n_repeats, n_thetas, n_data)
        sig = samples.std(axis=0, ddof=0).mean(axis=0)
        ref = np.abs(samples).mean(axis=(0, 1))
        sig = np.maximum(sig, float(floor) * np.maximum(ref, floor))
        return np.asarray(sig, dtype=float)

    def stats_sigma(self) -> np.ndarray:
        if self._stats_sigma is None:
            self._stats_sigma = self.estimate_stats_sigma()
        return self._stats_sigma

    def log_likelihood(self, x: ArrayLike, theta: ArrayLike) -> np.ndarray:
        """Gaussian log-likelihood of the summary statistics (see module docstring)."""
        x_arr = _as_2d_x(x)
        mean = self.data_mean(theta)  # (n_theta, n_data)
        sigma = self.stats_sigma()
        if mean.shape[-1] != x_arr.shape[-1]:
            raise ValueError(
                f"data dimension mismatch: x has {x_arr.shape[-1]}, expected {mean.shape[-1]}"
            )
        n_out = max(mean.shape[0], x_arr.shape[0])
        mean_b = np.broadcast_to(mean, (n_out, mean.shape[-1]))
        x_b = np.broadcast_to(x_arr, (n_out, x_arr.shape[-1]))
        return _diag_gaussian_logpdf(x_b, mean_b, sigma)

    def log_joint(self, theta: ArrayLike, x: ArrayLike) -> np.ndarray:
        return self.log_prior(theta) + self.log_likelihood(x, theta)

    def posterior_log_prob(
        self, theta: ArrayLike, x_obs: ArrayLike, *, normalize: bool = True
    ) -> np.ndarray:
        lp = self.log_joint(theta, x_obs)
        if not normalize:
            return lp
        finite = np.isfinite(lp)
        if not np.any(finite):
            return lp
        return lp - float(np.max(lp[finite]))

    ground_truth_log_posterior = posterior_log_prob

    def map_estimate(
        self,
        x_obs: ArrayLike,
        *,
        n_samples: int = 2000,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rng = _rng(rng, seed)
        cand = self.prior_sample(int(n_samples), rng)
        lp = self.posterior_log_prob(cand, _as_2d_x(x_obs)[0])
        return cand[int(np.argmax(lp))]

    # -- reference posterior (MCMC) ---------------------------------------
    def _mh_reference(
        self,
        x_obs: ArrayLike,
        n_samples: int,
        rng: np.random.Generator,
        *,
        n_steps: Optional[int] = None,
        step_size: Optional[float] = None,
    ) -> np.ndarray:
        x_obs = _as_2d_x(x_obs)[0]
        low, high = self.prior_bounds()
        n_steps = int(self.config.reference_steps if n_steps is None else n_steps)
        if step_size is None:
            step_size = self.config.reference_step_size
        if step_size is None:
            step_size = 0.15 * float(np.mean(high - low)) / math.sqrt(low.size)

        chains = self.prior_sample(int(n_samples), rng)
        logp = self.posterior_log_prob(chains, x_obs)
        logp = np.nan_to_num(logp, nan=-np.inf, posinf=0.0, neginf=-np.inf)
        accepted = np.zeros(chains.shape[0], dtype=bool)
        n_acc = 0
        for step in range(n_steps):
            prop = chains + float(step_size) * rng.standard_normal(chains.shape)
            in_box = np.all((prop >= low) & (prop <= high), axis=-1)
            logp_prop = np.full(chains.shape[0], -np.inf)
            if np.any(in_box):
                logp_prop[in_box] = self.posterior_log_prob(prop[in_box], x_obs)
            with np.errstate(invalid="ignore"):
                log_ratio = logp_prop - logp
            accept = np.log(rng.uniform(size=chains.shape[0])) < log_ratio
            accept &= np.isfinite(log_ratio)
            chains[accept] = prop[accept]
            logp[accept] = logp_prop[accept]
            n_acc += int(np.sum(accept))
            if (step + 1) % 50 == 0 and step + 1 < n_steps:
                rate = n_acc / (50.0 * chains.shape[0])
                if rate < 0.25:
                    step_size *= 0.8
                elif rate > 0.45:
                    step_size *= 1.2
                n_acc = 0
        return chains

    def reference_posterior_sample(
        self,
        x_obs: ArrayLike,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        *,
        seed: Optional[int] = None,
        burn_in: Optional[int] = None,
        thinning: int = 1,
        use_mcmc_module: bool = True,
        n_steps: Optional[int] = None,
        step_size: Optional[float] = None,
    ) -> np.ndarray:
        """Ground-truth posterior samples ``p(theta | x_obs)``."""
        del burn_in, thinning
        rng = _rng(rng, seed)
        if use_mcmc_module:
            try:  # prefer the shared reference MCMC implementation
                from ..reference.mcmc import sample_reference  # type: ignore

                mask = np.concatenate(
                    [np.zeros(self.n_parameters), np.ones(self.n_data)]
                )
                values = np.concatenate(
                    [np.zeros(self.n_parameters), np.asarray(_as_2d_x(x_obs)[0], dtype=float)]
                )
                out = sample_reference(
                    self,
                    mask,
                    values,
                    int(n_samples),
                    rng=rng,
                    return_full=False,
                    n_steps=n_steps,
                    step_size=step_size,
                )
                out = np.asarray(out, dtype=float)
                if out.ndim == 2 and out.shape[-1] == self.n_parameters:
                    return out
            except Exception:
                pass
        return self._mh_reference(
            x_obs, int(n_samples), rng, n_steps=n_steps, step_size=step_size
        )

    ground_truth_posterior = reference_posterior_sample

    def posterior_mean_std(
        self,
        x_obs: ArrayLike,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples = self.reference_posterior_sample(
            x_obs, n_samples=int(n_samples), rng=rng, **kwargs
        )
        return samples.mean(axis=0), samples.std(axis=0)

    # -- joint / datasets --------------------------------------------------
    def to_joint(self, theta: ArrayLike, x: ArrayLike) -> np.ndarray:
        theta_arr = _as_2d_theta(theta)
        x_arr = _as_2d_x(x)
        if x_arr.shape[0] == 1 and theta_arr.shape[0] > 1:
            x_arr = np.broadcast_to(x_arr, (theta_arr.shape[0], x_arr.shape[-1]))
        if theta_arr.shape[0] == 1 and x_arr.shape[0] > 1:
            theta_arr = np.broadcast_to(theta_arr, (x_arr.shape[0], theta_arr.shape[-1]))
        return np.concatenate([theta_arr, x_arr], axis=-1)

    def split_joint(self, joint: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        joint = np.atleast_2d(np.asarray(joint, dtype=float))
        return joint[:, : self.n_parameters], joint[:, self.n_parameters:]

    def sample_joint(
        self, n_samples: int = 1, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        rng = _rng(rng)
        theta, x = self(int(n_samples), rng)
        return self.to_joint(theta, x)

    def make_dataset(
        self,
        n_simulations: int,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        chunk_size: int = 256,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate ``n_simulations`` parameter/data pairs."""
        rng = _rng(rng, seed)
        n_simulations = int(n_simulations)
        thetas: List[np.ndarray] = []
        xs: List[np.ndarray] = []
        chunk = max(int(chunk_size), 1)
        done = 0
        while done < n_simulations:
            size = min(chunk, n_simulations - done)
            theta = self.prior_sample(size, rng)
            x = self.simulate(theta, rng)
            thetas.append(theta)
            xs.append(x)
            done += size
            if verbose:
                print(f"[hodgkin_huxley] simulated {done}/{n_simulations}", flush=True)
        return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0)

    def make_conditional_dataset(
        self,
        n_simulations: int,
        *,
        include_energy_stat: bool = True,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Joint dataset, optionally without the energy statistic."""
        rng = _rng(rng, seed)
        theta = self.prior_sample(int(n_simulations), rng)
        x = self.simulate(theta, rng, include_energy=include_energy_stat)
        return self.to_joint(theta, x)

    # -- energy / observation intervals (Sec. 4.4, Fig. 7) ----------------
    def energy(
        self, theta: ArrayLike, *, add_noise: bool = False,
        rng: Optional[np.random.Generator] = None, seed: Optional[int] = None,
    ) -> np.ndarray:
        """Metabolic cost (uJ/s) of the simulated trajectories for ``theta``."""
        rng = _rng(rng, seed)
        theta_arr = _as_2d_theta(theta)
        values = []
        for th in theta_arr:
            traj, _ = self.trajectory(th, add_noise=add_noise, rng=rng)
            values.append(
                energy_from_trajectory(
                    traj,
                    duration=self.config.duration,
                    area_cm2=self.config.energy_area_cm2,
                    j_per_coulomb=self.config.j_per_coulomb,
                )
            )
        return np.asarray(values, dtype=float)

    def posterior_predictive(
        self,
        theta_samples: ArrayLike,
        *,
        add_noise: bool = True,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Summary-statistic posterior predictives for posterior parameter samples."""
        return self.simulate(theta_samples, _rng(rng, seed), add_noise=add_noise)

    def energy_from_statistics(self, x: ArrayLike) -> np.ndarray:
        """Extract the energy statistic from a data/statistic array."""
        if self.energy_index is None:
            raise ValueError("energy statistic is disabled for this task")
        x_arr = _as_2d_x(x)
        return x_arr[:, self.energy_index]

    def energy_quantile_threshold(
        self, energy_samples: ArrayLike, quantile: float = 0.1
    ) -> float:
        """Lowest-``quantile`` energy threshold used for the interval constraint."""
        e = np.asarray(energy_samples, dtype=float).reshape(-1)
        return float(np.quantile(e, float(quantile)))

    def energy_interval(
        self,
        energy_samples: ArrayLike,
        *,
        quantile: float = 0.1,
        lower: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Interval ``[lower, q_quantile]`` on the energy statistic (Sec. 4.4)."""
        upper = self.energy_quantile_threshold(energy_samples, quantile)
        lo = float(lower) if lower is not None else -np.inf
        return lo, upper

    def energy_constraint(
        self,
        energy_samples: ArrayLike,
        *,
        quantile: float = 0.1,
        lower: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Descriptor for the energy observation interval used with guidance.

        The returned dict is directly consumable by
        :func:`simformer.guidance.interval_constraint`:
        ``indices`` (energy statistic index), ``lower`` and ``upper``.
        """
        lo, up = self.energy_interval(energy_samples, quantile=quantile, lower=lower)
        if self.energy_index is None:
            raise ValueError("energy statistic is disabled for this task")
        return {
            "indices": np.array([self.energy_index], dtype=int),
            "lower": np.array([lo], dtype=float),
            "upper": np.array([up], dtype=float),
            "quantile": float(quantile),
        }

    def observation_condition_mask(
        self, *, include_energy: Optional[bool] = None
    ) -> np.ndarray:
        """Condition mask for posterior inference on the summary statistics."""
        include = self.config.include_energy if include_energy is None else bool(include_energy)
        n_stats = len(DEFAULT_STAT_NAMES) + (1 if include else 0)
        return np.concatenate(
            [np.zeros(self.n_parameters), np.ones(n_stats)]
        ).astype(float)

    def posterior_condition_mask(self, include_energy: Optional[bool] = None) -> np.ndarray:
        return self.observation_condition_mask(include_energy=include_energy)

    def likelihood_condition_mask(self, *, include_energy: Optional[bool] = None) -> np.ndarray:
        include = self.config.include_energy if include_energy is None else bool(include_energy)
        n_stats = len(DEFAULT_STAT_NAMES) + (1 if include else 0)
        return np.concatenate(
            [np.ones(self.n_parameters), np.zeros(n_stats)]
        ).astype(float)

    def joint_condition_mask(self) -> np.ndarray:
        return np.zeros(self.n_parameters + self.n_data, dtype=float)

    def figure7_scenarios(
        self,
        *,
        n_simulations: int = 1000,
        quantile: float = 0.1,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Reference scenario definitions for the Sec. 4.4 (Fig. 7) experiment.

        Returns a dict with

        * ``observation``: the summary statistics used as conditioning data,
        * ``voltage_mask``: condition mask for voltage-summary-statistics only
          (no energy statistic),
        * ``energy_interval``: the interval constraint on energy derived from
          posterior-predictive samples (lowest ``quantile``),
        * ``energy_samples``: the posterior-predictive energy values.
        """
        rng = _rng(rng, seed)
        theta = self.prior_sample(int(n_simulations), rng)
        x = self.simulate(theta, rng)
        obs = x[0]
        energy_samples = x[:, self.energy_index] if self.energy_index is not None else None
        energy_interval = (
            self.energy_interval(energy_samples, quantile=quantile)
            if energy_samples is not None
            else None
        )
        return {
            "observation": obs,
            "voltage_observation": (
                obs[: len(DEFAULT_STAT_NAMES)] if energy_samples is not None else obs
            ),
            "voltage_mask": self.observation_condition_mask(include_energy=False),
            "energy_mask": self.energy_interval(energy_samples, quantile=quantile)
            if energy_samples is not None
            else None,
            "energy_samples": energy_samples,
            "energy_interval": energy_interval,
        }

    # -- Simformer plumbing ------------------------------------------------
    def spec(self, token_dim: int = 50, **kwargs: Any):
        """Tokenizer specification for the joint vector ``[theta (7) | stats]``."""
        try:
            from ..tokenizer import build_benchmark_spec  # type: ignore
        except Exception:  # pragma: no cover - import fallback
            from simformer.tokenizer import build_benchmark_spec  # type: ignore

        try:
            return build_benchmark_spec(self.n_parameters, self.n_data)
        except TypeError:  # pragma: no cover - alternate signature
            return build_benchmark_spec(
                n_parameters=self.n_parameters, n_data=self.n_data
            )

    token_spec = spec

    def _fallback_mask(self, *, directed: bool = True) -> np.ndarray:
        """Structured mask: factorized parameters, dense statistics block.

        Each summary statistic depends on all parameters (dense theta -> data
        block) and statistics are correlated with each other (dense data
        block); the parameters are independent a priori (identity block).
        """
        n_p, n_d = self.n_parameters, self.n_data
        mask = np.zeros((n_p + n_d, n_p + n_d), dtype=bool)
        mask[:n_p, :n_p] = np.eye(n_p, dtype=bool)
        mask[:n_p, n_p:] = True          # each statistic depends on all params
        mask[n_p:, n_p:] = True          # statistics are jointly dependent
        np.fill_diagonal(mask, True)
        if not directed:
            mask = mask | mask.T
        return mask

    def attention_mask(self, directed: bool = True, **kwargs: Any) -> np.ndarray:
        """Task attention mask ``M_E`` from :mod:`simformer.attention_masks`."""
        n_tokens = self.n_parameters + self.n_data
        try:
            try:
                from ..attention_masks import build_attention_mask  # type: ignore
            except Exception:  # pragma: no cover - import fallback
                from simformer.attention_masks import build_attention_mask  # type: ignore

            attempts = (
                dict(n_theta=self.n_parameters, n_x=self.n_data, n_stats=self.n_data,
                     n_series=1, n_times=0, directed=directed),
                dict(n_theta=self.n_parameters, n_x=self.n_data, directed=directed),
                dict(n_theta=self.n_parameters, n_x=self.n_data),
            )
            for kw in attempts:
                try:
                    mask = np.asarray(
                        build_attention_mask("hodgkin_huxley", **kw), dtype=bool
                    )
                except Exception:
                    continue
                if mask.shape == (n_tokens, n_tokens):
                    np.fill_diagonal(mask, True)
                    if not directed:
                        mask = mask | mask.T
                    return mask
        except Exception:
            pass
        return self._fallback_mask(directed=directed)

    def build_tokenizer(self, token_dim: int = 50, **kwargs: Any):
        try:
            try:
                from ..tokenizer import Tokenizer  # type: ignore
            except Exception:  # pragma: no cover
                from simformer.tokenizer import Tokenizer  # type: ignore

            return Tokenizer(self.spec(token_dim=token_dim), token_dim=token_dim)
        except Exception:
            return None

    def build_model(self, **kwargs: Any):
        """Build a Simformer score network wired to the HH tokenizer and mask."""
        try:
            try:
                from ..transformer import build_score_network  # type: ignore
            except Exception:  # pragma: no cover
                from simformer.transformer import build_score_network  # type: ignore

            kwargs.setdefault("task", "hodgkin_huxley")
            tokenizer = self.build_tokenizer(token_dim=int(kwargs.get("token_dim", 50)))
            candidates = [
                dict(task="hodgkin_huxley", spec=self.spec(), tokenizer=tokenizer,
                     attention_mask=self.attention_mask(directed=True), **kwargs),
                dict(task="hodgkin_huxley", tokenizer=tokenizer,
                     attention_mask=self.attention_mask(directed=True), **kwargs),
                dict(task="hodgkin_huxley", spec=self.spec(), **kwargs),
            ]
            for kw in candidates:
                try:
                    return build_score_network(**kw)
                except TypeError:
                    continue
            return build_score_network(spec=self.spec(), **kwargs)
        except Exception:
            return None


# -- aliases ---------------------------------------------------------------
Task = HodgkinHuxleyTask
Simulator = HodgkinHuxleyTask
HodgkinHuxley = HodgkinHuxleyTask


# ---------------------------------------------------------------------------
# Factory and module-level convenience API
# ---------------------------------------------------------------------------

def build_task(
    config: Optional[Union[HodgkinHuxleyConfig, Dict[str, Any]]] = None, **kwargs: Any
) -> HodgkinHuxleyTask:
    """Instantiate the Hodgkin-Huxley task."""
    return HodgkinHuxleyTask(config, **kwargs)


_DEFAULT_TASK: Optional[HodgkinHuxleyTask] = None


def _default_task() -> HodgkinHuxleyTask:
    global _DEFAULT_TASK
    if _DEFAULT_TASK is None:
        _DEFAULT_TASK = HodgkinHuxleyTask()
    return _DEFAULT_TASK


def prior_sample(
    n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any
) -> np.ndarray:
    return _default_task().prior_sample(n_samples, rng)


def log_prior(theta: ArrayLike, **kwargs: Any) -> np.ndarray:
    return _default_task().log_prior(theta)


def simulate(
    theta: ArrayLike, rng: Optional[np.random.Generator] = None, **kwargs: Any
) -> np.ndarray:
    return _default_task().simulate(theta, rng, **kwargs)


def log_likelihood(x: ArrayLike, theta: ArrayLike, **kwargs: Any) -> np.ndarray:
    return _default_task().log_likelihood(x, theta)


def log_joint(theta: ArrayLike, x: ArrayLike, **kwargs: Any) -> np.ndarray:
    return _default_task().log_joint(theta, x)


def posterior_log_prob(
    theta: ArrayLike, x_obs: ArrayLike, **kwargs: Any
) -> np.ndarray:
    return _default_task().posterior_log_prob(theta, x_obs, **kwargs)


def reference_posterior_sample(
    x_obs: ArrayLike, n_samples: int = 1000, rng: Optional[np.random.Generator] = None,
    **kwargs: Any
) -> np.ndarray:
    return _default_task().reference_posterior_sample(x_obs, n_samples, rng, **kwargs)


def make_dataset(
    n_simulations: int,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    chunk_size: int = 256,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    return _default_task().make_dataset(
        n_simulations, rng=rng, seed=seed, chunk_size=chunk_size, verbose=verbose
    )


__all__ = [
    # config / task
    "HodgkinHuxleyConfig",
    "HodgkinHuxleyTask",
    "HodgkinHuxley",
    "Task",
    "Simulator",
    "build_task",
    # module-level API
    "prior_sample",
    "log_prior",
    "simulate",
    "log_likelihood",
    "log_joint",
    "posterior_log_prob",
    "reference_posterior_sample",
    "make_dataset",
    # dynamics / rate functions
    "efun",
    "alpha_m",
    "beta_m",
    "alpha_h",
    "beta_h",
    "alpha_n",
    "beta_n",
    "steady_state_gating",
    "gating_derivatives",
    "stimulus_current",
    "hodgkin_huxley_rhs",
    "integrate_hodgkin_huxley",
    "integrate_trajectories",
    # statistics / energy
    "spike_times",
    "spike_count",
    "central_moments",
    "summary_statistics",
    "hodgkin_huxley_features",
    "sodium_charge",
    "convert_charge_to_energy",
    "energy_from_trajectory",
    # constants
    "DEFAULT_PARAMETER_NAMES",
    "DEFAULT_PRIOR_RANGES",
    "DEFAULT_STAT_NAMES",
    "ENERGY_STAT_NAME",
    "DEFAULT_DURATION",
    "DEFAULT_DT",
    "DEFAULT_V0",
    "DEFAULT_I_INJ",
    "DEFAULT_NOISE_SIGMA",
    "FARADAY_CONSTANT",
    "ATP_ENERGY_J_PER_MOL",
    "NA_IONS_PER_ATP",
    "J_PER_COULOMB",
]
