"""Hodgkin-Huxley task of Sec. 4.4 (Appendix A2.2).

We follow the implementation guidelines of Pospischil et al. (2008): the
membrane voltage starts at ``V0 = -65 mV``, a current of ``4`` is injected
between ``50 ms`` and ``150 ms`` and the simulation runs for ``200 ms`` with a
stochastic term ``0.05 dW_t``.  The voltage trace is reduced to the summary
statistics of Gonçalves et al. (2020) and -- additionally -- to the metabolic
energy consumption that is computed from the sodium charge
(``convert_charge_to_energy`` of the addendum).

The joint distribution of the Simformer therefore consists of the 7 parameters,
the 7 voltage summary statistics and 1 additional energy statistic (the latter
is what the guided diffusion experiment of Fig. 7e/f constrains).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..masks import dense_mask
from .base import Task


# --------------------------------------------------------------------------- #
#  Rate functions (addendum)
# --------------------------------------------------------------------------- #
def efun(x):
    """``efun(x) = 1 - x/2`` for ``x < 1e-4`` and ``x / (exp(x) - 1)`` else."""
    x = np.asarray(x, dtype=float)
    scalar = (x.ndim == 0)
    x = np.atleast_1d(x)
    out = np.empty_like(x)
    small = x < 1e-4
    out[small] = 1.0 - 0.5 * x[small]
    rest = x[~small]
    out[~small] = rest / np.expm1(rest)
    return out[0] if scalar else out


def alpha_m(V, V0=-65.0):
    return 0.32 * efun(-0.25 * (V - V0 - 13.0)) / 0.25


def beta_m(V, V0=-65.0):
    return 0.28 * efun(0.2 * (V - V0 - 40.0)) / 0.2


def alpha_h(V, V0=-65.0):
    return 0.128 * np.exp(-(V - V0 - 17.0) / 18.0)


def beta_h(V, V0=-65.0):
    return 4.0 / (1.0 + np.exp(-(V - V0 - 40.0) / 5.0))


def alpha_n(V, V0=-65.0):
    return 0.032 * efun(-0.2 * (V - V0 - 15.0)) / 0.2


def beta_n(V, V0=-65.0):
    return 0.5 * np.exp(-(V - V0 - 10.0) / 40.0)


# --------------------------------------------------------------------------- #
#  Energy consumption (addendum, "convert_charge_to_energy")
# --------------------------------------------------------------------------- #
def convert_total_energy(E):
    E = -E                    # energy is negative
    E = E / 1000              # mS to S
    E = E / 1000              # mV to V
    E = E * 0.628e-3          # area of the membrane
    e = 1.602176634e-19       # elementary charge
    N_Na = E / e              # number of elementary charges
    valence_Na = 1            # valence of sodium
    number_of_transports = 3  # number of Na out per ATP
    ATP_Na = N_Na / (valence_Na * number_of_transports)
    ATP_energy = 10e-19       # energy by ATP hydrolysis
    E = ATP_Na * ATP_energy   # energy in joules
    E = E / 0.2               # energy in J/s
    return E * 1e6            # energy in uJ/s


def convert_charge_to_energy(E):
    """Convert the (cumulative, per time step) sodium charge into energy.

    ``E`` has shape ``(..., n_steps)`` and holds the cumulative sodium charge of
    the trace; the function returns the smoothed energy per time step in
    ``uJ/s`` (exactly as in the addendum).
    """
    E = np.asarray(E, dtype=float)
    if E.ndim == 1:
        E = E[None, :]
    dE = np.diff(E, axis=-1)             # non cumulative energy
    kernel = np.ones(5) / 5.0
    if dE.shape[-1] >= kernel.size:
        dE = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), -1, dE)
    return convert_total_energy(dE)


# --------------------------------------------------------------------------- #
#  Simulator
# --------------------------------------------------------------------------- #
def simulate_hodgkin_huxley(theta: np.ndarray, dt: float = 0.01,
                            duration: float = 200.0,
                            i_inj_amplitude: float = 4.0,
                            i_inj_start: float = 50.0,
                            i_inj_end: float = 150.0,
                            v0: float = -65.0,
                            noise_std: float = 0.05,
                            rng: Optional[np.random.Generator] = None):
    """Simulate the stochastic Hodgkin-Huxley model for a batch of parameters.

    Parameters (7): ``g_Na, g_K, g_L, E_Na, E_K, E_L, C_m``.

    Returns
    -------
    voltage : ``(n, n_steps)`` voltage trace
    energy : ``(n,)`` sodium based energy consumption in ``uJ/s``
    """
    theta = np.atleast_2d(theta)
    n = theta.shape[0]
    rng = rng if rng is not None else np.random.default_rng()
    g_na, g_k, g_l = theta[:, 0], theta[:, 1], theta[:, 2]
    e_na, e_k, e_l = theta[:, 3], theta[:, 4], theta[:, 5]
    c_m = theta[:, 6]
    n_steps = int(round(duration / dt)) + 1
    times = np.linspace(0.0, duration, n_steps)

    V = np.full(n, v0)
    # steady state initial conditions of the gating variables
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    nn = alpha_n(V) / (alpha_n(V) + beta_n(V))

    voltage = np.empty((n, n_steps))
    charge_cumulative = np.zeros(n)
    charge = np.empty((n, n_steps))
    sqrt_dt = np.sqrt(dt)
    for i, t in enumerate(times):
        voltage[:, i] = V
        i_inj = i_inj_amplitude if i_inj_start <= t <= i_inj_end else 0.0
        i_na = g_na * m ** 3 * h * (V - e_na)
        i_k = g_k * nn ** 4 * (V - e_k)
        i_l = g_l * (V - e_l)
        dV = (i_inj - i_na - i_k - i_l) / c_m
        # integrate the sodium charge in *seconds* (see addendum)
        charge_cumulative += i_na * dt * 1e-3
        charge[:, i] = charge_cumulative
        noise = noise_std * sqrt_dt * rng.normal(size=n)
        # clip the voltage / gating variables to physical ranges: for extreme
        # parameter sets the explicit Euler-Maruyama step can diverge, which
        # would otherwise produce NaN summary statistics.
        V = np.clip(V + dt * dV + noise, -120.0, 100.0)
        a_m, b_m = alpha_m(V), beta_m(V)
        a_h, b_h = alpha_h(V), beta_h(V)
        a_n, b_n = alpha_n(V), beta_n(V)
        m = np.clip(m + dt * (a_m * (1 - m) - b_m * m), 0.0, 1.0)
        h = np.clip(h + dt * (a_h * (1 - h) - b_h * h), 0.0, 1.0)
        nn = np.clip(nn + dt * (a_n * (1 - nn) - b_n * nn), 0.0, 1.0)

    # `convert_charge_to_energy` returns the (smoothed) energy per time step; the
    # energy consumption of the trace is their sum (in ``uJ/s``).
    energy = convert_charge_to_energy(charge).sum(axis=-1)
    return voltage, energy


def summary_statistics(voltage: np.ndarray, times: np.ndarray,
                       i_inj_start: float = 50.0, i_inj_end: float = 150.0,
                       spike_threshold: float = 0.0):
    """The seven summary statistics of the addendum."""
    resting = (times < i_inj_start)
    spiking = (times >= i_inj_start) & (times <= i_inj_end)
    resting_v = voltage[:, resting]
    spiking_v = voltage[:, spiking]

    above = spiking_v > spike_threshold
    crossings = np.sum((~above[:, :-1]) & above[:, 1:], axis=-1).astype(float)

    mean_rest = resting_v.mean(axis=-1)
    std_rest = resting_v.std(axis=-1)
    mean_spike = spiking_v.mean(axis=-1)
    centered = spiking_v - mean_spike[:, None]
    m2 = (centered ** 2).mean(axis=-1)
    m3 = (centered ** 3).mean(axis=-1)
    m4 = (centered ** 4).mean(axis=-1)
    return np.stack([crossings, mean_rest, std_rest, mean_spike, m2, m3, m4],
                    axis=-1)


class HodgkinHuxleyTask(Task):
    """Hodgkin-Huxley with voltage summary statistics and energy consumption."""

    name = "hodgkin_huxley"
    n_params = 7
    n_voltage_stats = 7
    n_data = 8                       # 7 voltage statistics + energy
    param_names = ["g_Na", "g_K", "g_L", "E_Na", "E_K", "E_L", "C_m"]
    # Uniform priors centred on the Hodgkin-Huxley model of Pospischil et al.
    # (2008), which the paper refers to for the implementation guidelines.  (The
    # paper does not restate the prior ranges, these are the ranges we use:
    # parameters are (g_Na, g_K, g_L, E_Na, E_K, E_L, C_m).)
    prior_low = np.array([10.0, 3.0, 0.005, 40.0, -95.0, -80.0, 0.5])
    prior_high = np.array([120.0, 20.0, 0.05, 60.0, -75.0, -60.0, 2.0])

    def __init__(self, dt: float = 0.01, duration: float = 200.0, seed: int = 0):
        self.dt = dt
        self.duration = duration
        self.times = np.linspace(0.0, duration, int(round(duration / dt)) + 1)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ prior
    def prior_sample(self, n, rng):
        return rng.uniform(self.prior_low, self.prior_high, size=(n, self.n_params))

    def log_prior(self, theta):
        theta = np.atleast_2d(theta)
        inside = np.all((theta >= self.prior_low) & (theta <= self.prior_high),
                        axis=-1)
        volume = np.prod(self.prior_high - self.prior_low)
        return np.where(inside, -np.log(volume), -np.inf)

    # -------------------------------------------------------------- simulation
    def simulate(self, theta, rng):
        voltage, energy = simulate_hodgkin_huxley(theta, dt=self.dt,
                                                  duration=self.duration,
                                                  rng=rng)
        stats = summary_statistics(voltage, self.times)
        return np.concatenate([stats, energy[:, None]], axis=-1)

    def simulate_trace(self, theta, rng=None):
        """Return ``(voltage, energy, statistics)`` for a batch of parameters."""
        voltage, energy = simulate_hodgkin_huxley(
            theta, dt=self.dt, duration=self.duration,
            rng=rng if rng is not None else self._rng)
        stats = summary_statistics(voltage, self.times)
        return voltage, energy, stats

    # -------------------------------------------------------------- structure
    def base_mask(self):
        """Every summary statistic depends on all parameters; the parameters are
        a priori independent."""
        n = self.n_variables
        mask = np.zeros((n, n), dtype=bool)
        mask[:self.n_params, :self.n_params] = np.eye(self.n_params, dtype=bool)
        mask[self.n_params:, :self.n_params] = True
        mask[self.n_params:, self.n_params:] = True   # statistics are coupled
        return mask

    def variable_kind(self):
        kinds = np.zeros(self.n_variables, dtype=np.int64)
        kinds[1:] = 1
        return kinds

    def problem(self):
        return super().problem()
