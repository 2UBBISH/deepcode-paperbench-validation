"""Gravitational-wave benchmark task (Hermans et al. 2022 style).

Implements the task described in Appendix A3.2 of *Simformer: Simulation-based
inference with probabilistic diffusion models*:

    "*As a test case, we will consider the Gravitational Waves benchmark tasks as
    presented in Hermans et al. (2022). In this case, we have low dimensional
    :math:`\\theta \\in \\mathbb{R}^2`, i.e., the masses of the two black holes,
    and two high dimensional :math:`x \\in \\mathbb{R}^{8192}` measurements of the
    corresponding gravitational waves from two different detectors. ... we only
    target the conditionals :math:`p(\\theta \\mid x_1, x_2)`,
    :math:`p(\\theta \\mid x_1)` and :math:`p(\\theta \\mid x_2)`. We use 100k
    simulations for training.*"

Because the exact waveform/injection model of Hermans et al. (2022) is not
reproduced verbatim in the paper, this module implements a self-contained,
physically motivated reduced-order inspiral model:

* two component masses ``m1, m2`` (in solar masses) are drawn uniformly from
  ``[10, 80]`` (a standard binary-black-hole prior; consistent with the ~20 Hz
  inspiral band visible in a one second segment for these masses),
* the strain time series follows the leading-order (Newtonian) chirp
  ``f(tau) = (5/256)^{3/8} / (pi * M_c) * tau^{-3/8}`` with chirp mass
  ``M_c = (m1 m2)^{3/5} / (m1 + m2)^{1/5}`` and amplitude
  ``A ∝ M_c^{5/3} f^{2/3}``,
* both detectors observe the *same* source with detector-specific response
  factors and independent Gaussian noise realisations, which makes the two
  measurements (and hence the partial posteriors) statistically dependent.

The module exposes the simulator, the prior, targeted (partial) posterior
condition masks and the tokenizer/model factories used by the training scripts.
Only ``numpy`` is required; ``torch`` is imported lazily for the model factory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Geometrised solar mass, in seconds (T_sun = G M_sun / c^3).
SOLAR_MASS_TIME: float = 4.925490947e-6
#: Reference solar mass in kg, used only for documentation/dimensional checks.
SOLAR_MASS_KG: float = 1.98892e30

#: Dimension of each detector measurement (Appendix A3.2).
GW_MEASUREMENT_DIM: int = 8192
#: Number of detectors / measurement blocks.
GW_N_DETECTORS: int = 2
#: Number of parameters (black-hole masses).
GW_N_PARAMETERS: int = 2
#: Partial posteriors that are targeted during training (Appendix A3.2).
GW_TARGETS: Tuple[str, ...] = ("x1_x2", "x1", "x2")

#: Default uniform prior bounds on each component mass (solar masses).
DEFAULT_MASS_MIN: float = 10.0
DEFAULT_MASS_MAX: float = 80.0
#: Duration of the analysed segment (seconds) and its sampling frequency (Hz).
DEFAULT_DURATION: float = 1.0
DEFAULT_SAMPLE_RATE: float = float(GW_MEASUREMENT_DIM)  # 8192 Hz -> dt = 1/8192
#: Frequency band (Hz) where the detector is assumed to be sensitive.
DEFAULT_F_MIN: float = 20.0
#: Detector response factors (antenna patterns) for the two interferometers.
DEFAULT_DETECTOR_RESPONSE: Tuple[float, ...] = (1.0, 0.75)
#: Detector-specific time delays (seconds) w.r.t. the reference detector.
DEFAULT_DETECTOR_DELAYS: Tuple[float, ...] = (0.0, 0.003)
#: Detector-specific phase shifts (radians).
DEFAULT_DETECTOR_PHASES: Tuple[float, ...] = (0.0, 0.6)
#: Standard deviation of the additive Gaussian detector noise.
DEFAULT_NOISE_SIGMA: float = 0.35
#: Numerical floor on ``tau`` to avoid diverging instantaneous frequency.
_TAU_FLOOR: float = 1e-4


# --------------------------------------------------------------------------- #
# Waveform model
# --------------------------------------------------------------------------- #


def chirp_mass(m1: float, m2: float) -> float:
    """Chirp mass of a binary, in solar masses.

    ``M_c = (m1 * m2)^{3/5} / (m1 + m2)^{1/5}``.
    """
    return float((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)


def symmetric_mass_ratio(m1: float, m2: float) -> float:
    """Symmetric mass ratio ``eta = m1 m2 / (m1 + m2)^2`` (in ``(0, 0.25]``)."""
    return float(m1 * m2 / (m1 + m2) ** 2)


def inspiral_frequency(
    tau: np.ndarray,
    chirp_mass_msun: float,
    f_min: float = DEFAULT_F_MIN,
    f_max: Optional[float] = None,
) -> np.ndarray:
    """Leading-order (Newtonian) inspiral frequency at time-to-coalescence ``tau``.

    Parameters
    ----------
    tau:
        Time *before* coalescence, in seconds (positive, decreases towards the
        merger).
    chirp_mass_msun:
        Chirp mass in solar masses.
    f_min, f_max:
        Frequency band; values are clipped to ``[f_min, f_max]`` when provided.
    """
    tau = np.clip(np.asarray(tau, dtype=np.float64), _TAU_FLOOR, None)
    mc_s = chirp_mass_msun * SOLAR_MASS_TIME
    f = (5.0 / 256.0) ** (3.0 / 8.0) / (math.pi * mc_s ** (5.0 / 8.0)) * tau ** (-3.0 / 8.0)
    if f_min is not None and f_min > 0:
        f = np.clip(f, f_min, None)
    if f_max is not None:
        f = np.clip(f, None, f_max)
    return f


def inspiral_waveform(
    m1: float,
    m2: float,
    *,
    n_samples: int = GW_MEASUREMENT_DIM,
    duration: float = DEFAULT_DURATION,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    f_min: float = DEFAULT_F_MIN,
    t_c: Optional[float] = None,
    phase0: float = 0.0,
    amplitude_scale: float = 1.0,
    start_time: Optional[float] = None,
    return_instantaneous_frequency: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Reduced-order time-domain inspiral strain for a binary of masses ``(m1, m2)``.

    The waveform is built by numerically integrating the instantaneous frequency
    (avoiding any stationary-phase approximation):

    .. math::

        h(t) = A(t) \\cos\\left(2\\pi \\int_{t}^{t_c} f(t') \\, dt' + \\phi_0\\right)

    with :math:`f(\\tau)` the Newtonian chirp frequency and
    :math:`A(t) \\propto M_c^{5/3} f(t)^{2/3}`.

    Returns the strain array of shape ``(n_samples,)`` (and optionally the
    instantaneous frequency array).
    """
    n_samples = int(n_samples)
    dt = 1.0 / float(sample_rate)
    if start_time is None:
        start_time = -n_samples * dt
    if t_c is None:
        t_c = 0.0
    times = start_time + dt * np.arange(n_samples, dtype=np.float64)
    tau = np.clip(t_c - times, _TAU_FLOOR, None)

    mc = chirp_mass(m1, m2)
    eta = symmetric_mass_ratio(m1, m2)
    nyquist = 0.5 * float(sample_rate)
    f = inspiral_frequency(tau, mc, f_min=0.0, f_max=nyquist)

    # Amplitude: Newtonian (quadrupole) scaling, normalised to O(1) strain.
    amp = amplitude_scale * mc ** (5.0 / 3.0) * (math.pi * f) ** (2.0 / 3.0)
    amp = amp * eta ** 0.0  # eta enters at higher PN order only
    scale = np.max(np.abs(amp)) if np.max(np.abs(amp)) > 0 else 1.0
    amp = amp / scale

    # Phase: numerically integrate 2 pi f. Outside the band (f < f_min) the
    # signal is band-limited, so we only accumulate phase where the detector is
    # sensitive and taper the amplitude to zero there.
    in_band = f >= f_min
    phase = phase0 + 2.0 * math.pi * np.cumsum(f) * dt
    strain = amp * np.cos(phase)
    strain = np.where(in_band, strain, 0.0)
    # Smooth (half-cosine) taper at the low-frequency edge.
    if np.any(in_band) and not np.all(in_band):
        taper = np.clip((f - 0.98 * f_min) / (f_min * 0.02), 0.0, 1.0)
        strain = strain * taper

    if return_instantaneous_frequency:
        return strain, f
    return strain


def project_to_detector(
    strain: np.ndarray,
    detector_index: int = 0,
    *,
    response: Optional[Sequence[float]] = None,
    delays: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Apply the detector response / time delay / phase shift to a strain series.

    The signal is shifted in time by the detector's light-travel delay (a
    sub-sample resampling in the frequency domain) and scaled by its antenna
    response factor.
    """
    strain = np.asarray(strain, dtype=np.float64)
    response = tuple(DEFAULT_DETECTOR_RESPONSE if response is None else response)
    delays = tuple(DEFAULT_DETECTOR_DELAYS if delays is None else delays)
    phases = tuple(DEFAULT_DETECTOR_PHASES if phases is None else phases)
    k = int(detector_index) % len(response)

    out = strain
    delay = float(delays[k]) if k < len(delays) else 0.0
    if abs(delay) > 0:
        n = out.shape[0]
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
        spectrum = np.fft.rfft(out)
        spectrum = spectrum * np.exp(-2j * math.pi * freqs * delay)
        out = np.fft.irfft(spectrum, n=n)
    phi = float(phases[k]) if k < len(phases) else 0.0
    if phi != 0.0:
        # Real phase shift via the analytic signal (Hilbert transform).
        analytic = np.fft.ifft(
            np.fft.fft(out)
            * np.where(np.fft.fftfreq(out.shape[0]) > 0, 2.0, np.where(np.fft.fftfreq(out.shape[0]) == 0, 1.0, 0.0))
        )
        out = np.real(analytic * np.exp(1j * phi))
    return response[k] * out


# --------------------------------------------------------------------------- #
# Prior / likelihood
# --------------------------------------------------------------------------- #


def sample_prior(
    n_samples: int = 1,
    rng: Optional[np.random.Generator] = None,
    *,
    mass_min: float = DEFAULT_MASS_MIN,
    mass_max: float = DEFAULT_MASS_MAX,
    sort_masses: bool = False,
) -> np.ndarray:
    """Uniform prior over component masses; returns ``(n_samples, 2)`` in solar masses."""
    rng = np.random.default_rng() if rng is None else rng
    theta = rng.uniform(mass_min, mass_max, size=(int(n_samples), 2))
    if sort_masses:
        theta = np.sort(theta, axis=-1)
    return theta.astype(np.float64)


def log_prior(
    theta: np.ndarray,
    *,
    mass_min: float = DEFAULT_MASS_MIN,
    mass_max: float = DEFAULT_MASS_MAX,
) -> np.ndarray:
    """Log-density of the (independent, uniform) mass prior."""
    theta = np.asarray(theta, dtype=np.float64)
    inside = np.all((theta >= mass_min) & (theta <= mass_max), axis=-1)
    log_norm = -np.log(mass_max - mass_min) * theta.shape[-1]
    logp = np.full(theta.shape[:-1], log_norm, dtype=np.float64)
    return np.where(inside, logp, -np.inf)


def log_likelihood_gaussian(
    x_obs: np.ndarray,
    x_mean: np.ndarray,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
) -> np.ndarray:
    """Independent Gaussian detector-noise likelihood, summed over measurements."""
    x_obs = np.asarray(x_obs, dtype=np.float64)
    x_mean = np.asarray(x_mean, dtype=np.float64)
    residual = x_obs - x_mean
    return -0.5 * np.sum(residual ** 2, axis=-1) / (noise_sigma ** 2)


# --------------------------------------------------------------------------- #
# Task configuration & simulator
# --------------------------------------------------------------------------- #


@dataclass
class GravitationalWavesConfig:
    """Configuration of the gravitational-wave simulator.

    Defaults follow Appendix A3.2: two parameters, two detectors with
    ``8192``-dimensional measurements, 100k training simulations.
    """

    n_parameters: int = GW_N_PARAMETERS
    measurement_dim: int = GW_MEASUREMENT_DIM
    n_detectors: int = GW_N_DETECTORS
    duration: float = DEFAULT_DURATION
    sample_rate: float = DEFAULT_SAMPLE_RATE
    mass_min: float = DEFAULT_MASS_MIN
    mass_max: float = DEFAULT_MASS_MAX
    f_min: float = DEFAULT_F_MIN
    detector_response: Tuple[float, ...] = DEFAULT_DETECTOR_RESPONSE
    detector_delays: Tuple[float, ...] = DEFAULT_DETECTOR_DELAYS
    detector_phases: Tuple[float, ...] = DEFAULT_DETECTOR_PHASES
    noise_sigma: float = DEFAULT_NOISE_SIGMA
    normalize_measurements: bool = False
    targets: Tuple[str, ...] = GW_TARGETS
    n_training_simulations: int = 100_000
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GravitationalWavesTask:
    """Simulator + prior + targeted-conditional helpers for the GW benchmark.

    Attributes
    ----------
    theta_dim:
        Number of scalar parameters (2 masses).
    data_dim:
        Number of *data tokens/variables*: when ``embedding`` is used each
        detector block is compressed into a single token, otherwise one variable
        per scalar measurement (``n_detectors * measurement_dim``).
    """

    def __init__(
        self,
        config: Optional[GravitationalWavesConfig] = None,
        *,
        embedding: bool = True,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = GravitationalWavesConfig(**{k: v for k, v in kwargs.items() if hasattr(GravitationalWavesConfig, k) or k in GravitationalWavesConfig.__dataclass_fields__})
        self.config = config
        self.embedding = bool(embedding)
        self.rng = np.random.default_rng(config.seed)

    # -- shapes ----------------------------------------------------------- #
    @property
    def theta_dim(self) -> int:
        return int(self.config.n_parameters)

    @property
    def measurement_dim(self) -> int:
        return int(self.config.measurement_dim)

    @property
    def n_detectors(self) -> int:
        return int(self.config.n_detectors)

    @property
    def data_dim(self) -> int:
        """Number of data tokens (1 per detector) or scalar data dimensions."""
        if self.embedding:
            return self.n_detectors
        return self.n_detectors * self.measurement_dim

    @property
    def targets(self) -> Tuple[str, ...]:
        return tuple(self.config.targets)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GravitationalWavesTask(theta_dim={self.theta_dim}, "
            f"n_detectors={self.n_detectors}, measurement_dim={self.measurement_dim}, "
            f"embedding={self.embedding})"
        )

    # -- prior ------------------------------------------------------------ #
    def prior_sample(
        self,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        return sample_prior(
            n_samples,
            rng if rng is not None else self.rng,
            mass_min=self.config.mass_min,
            mass_max=self.config.mass_max,
        )

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        return log_prior(theta, mass_min=self.config.mass_min, mass_max=self.config.mass_max)

    # -- simulator -------------------------------------------------------- #
    def _source_strain(self, theta: np.ndarray, t_c: float = 0.0) -> np.ndarray:
        return inspiral_waveform(
            float(theta[0]),
            float(theta[1]),
            n_samples=self.measurement_dim,
            duration=self.config.duration,
            sample_rate=self.config.sample_rate,
            f_min=self.config.f_min,
            t_c=t_c,
        )

    def simulate_measurements(
        self,
        theta: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        add_noise: bool = True,
        return_clean: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate detector measurements for a single parameter vector.

        Returns an array of shape ``(n_detectors, measurement_dim)``; if
        ``return_clean`` also returns the noise-free projected waveforms.
        """
        rng = self.rng if rng is None else rng
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        strain = self._source_strain(theta)
        clean = np.stack(
            [
                project_to_detector(
                    strain,
                    k,
                    response=self.config.detector_response,
                    delays=self.config.detector_delays,
                    phases=self.config.detector_phases,
                    sample_rate=self.config.sample_rate,
                )
                for k in range(self.n_detectors)
            ],
            axis=0,
        )
        if add_noise:
            noise = rng.normal(0.0, self.config.noise_sigma, size=clean.shape)
            obs = clean + noise
        else:
            obs = clean
        if self.config.normalize_measurements:
            obs = obs / self.config.noise_sigma
        if return_clean:
            return obs, clean
        return obs

    def simulate(
        self,
        n_samples: int = 1,
        theta: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        *,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Joint simulation; returns ``(theta, x)`` with ``x`` of shape ``(n, 2, D)``."""
        rng = self.rng if rng is None else rng
        if theta is None:
            theta = self.prior_sample(n_samples, rng)
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]
        observations = np.stack(
            [self.simulate_measurements(th, rng, add_noise=add_noise) for th in theta], axis=0
        )
        return theta, observations

    def __call__(
        self, n_samples: int = 1, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.simulate(n_samples, rng=rng)

    # -- joint-vector conventions ---------------------------------------- #
    def to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Flatten ``(theta, x)`` into the joint vector ``[theta, x_1, x_2]``."""
        theta = np.asarray(theta, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]
        if x.ndim == 2:
            x = x[None, ...]
        return np.concatenate([theta, x.reshape(x.shape[0], -1)], axis=-1)

    def split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split ``[theta, x_1, x_2]`` back into ``(theta, x)``."""
        joint = np.asarray(joint, dtype=np.float64)
        theta = joint[..., : self.theta_dim]
        data = joint[..., self.theta_dim :]
        x = data.reshape(*data.shape[:-1], self.n_detectors, self.measurement_dim)
        return theta, x

    @property
    def joint_dim(self) -> int:
        return self.theta_dim + self.n_detectors * self.measurement_dim

    # -- targeted (partial) posteriors ----------------------------------- #
    def active_blocks_for_target(self, target: str) -> List[int]:
        """Detector blocks that are observed for a targeted conditional.

        ``"x1_x2"`` -> ``[0, 1]`` (full posterior), ``"x1"`` -> ``[0]``,
        ``"x2"`` -> ``[1]``.
        """
        name = str(target).lower()
        if name in ("x1_x2", "x1x2", "both", "joint", "full"):
            return list(range(self.n_detectors))
        if name in ("x1", "1", "detector1", "h1"):
            return [0]
        if name in ("x2", "2", "detector2", "l1"):
            return [1]
        raise ValueError(f"Unknown gravitational-wave target {target!r}; expected one of {GW_TARGETS}.")

    def targeted_targets(self) -> Dict[str, List[int]]:
        """Mapping from target name to the observed detector blocks."""
        return {t: self.active_blocks_for_target(t) for t in self.targets}

    def condition_mask_for_target(self, target: str) -> np.ndarray:
        """Token-level condition mask ``M_C`` for a targeted conditional.

        Layout is ``[theta_1, theta_2, x_1, x_2]`` (one token per detector) with
        parameters latent (``0``) and the selected detector blocks observed
        (``1``).  With ``embedding=False`` the mask is expanded to the flattened
        measurement layout.
        """
        blocks = set(self.active_blocks_for_target(target))
        theta_part = np.zeros(self.theta_dim, dtype=np.float32)
        if self.embedding:
            data_part = np.array(
                [1.0 if k in blocks else 0.0 for k in range(self.n_detectors)], dtype=np.float32
            )
        else:
            data_part = np.concatenate(
                [
                    np.full(self.measurement_dim, 1.0 if k in blocks else 0.0, dtype=np.float32)
                    for k in range(self.n_detectors)
                ]
            )
        return np.concatenate([theta_part, data_part])

    def targeted_condition_masks(self) -> Dict[str, np.ndarray]:
        return {t: self.condition_mask_for_target(t) for t in self.targets}

    # -- tokenizer / model -------------------------------------------------- #
    def token_spec(self, token_dim: int = 50, embed_dim: int = 16):
        """Build the :class:`~simformer.embedding_nets.EmbeddingSpec` for this task."""
        from simformer.embedding_nets import EmbeddingSpec  # local import: optional torch

        return EmbeddingSpec(
            parameter_names=("m1", "m2"),
            data_dims=tuple([self.measurement_dim] * self.n_detectors),
            data_names=tuple(f"x{k + 1}" for k in range(self.n_detectors)),
            embed_dim=int(embed_dim),
            token_dim=int(token_dim),
            embedding_kind="cnn",
            share_embedding=True,
        )

    def build_tokenizer(self, token_dim: int = 50, embed_dim: int = 16):
        """Build the embedding tokenizer (CNN per detector block)."""
        from simformer.embedding_nets import EmbeddingTokenizer  # optional torch

        return EmbeddingTokenizer(self.token_spec(token_dim=token_dim, embed_dim=embed_dim))

    def build_model(self, **kwargs: Any):
        """Build a full Simformer score network for the GW task."""
        from simformer.embedding_nets import build_gravitational_waves_model

        params = dict(
            n_parameters=self.theta_dim,
            measurement_dim=self.measurement_dim,
            n_detectors=self.n_detectors,
        )
        params.update(kwargs)
        return build_gravitational_waves_model(**params)

    # -- datasets ---------------------------------------------------------- #
    def make_dataset(
        self,
        n_simulations: int = 100_000,
        *,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = False,
        chunk_size: int = 512,
    ) -> Dict[str, np.ndarray]:
        """Generate a training/evaluation dataset of ``n_simulations`` runs.

        Returns a dict with ``theta`` ``(n, 2)``, ``x`` ``(n, 2, D)``,
        ``joint`` ``(n, 2 + 2D)`` and the targeted condition masks.
        """
        rng = self.rng if rng is None else rng
        n_simulations = int(n_simulations)
        theta = self.prior_sample(n_simulations, rng)
        x = np.empty((n_simulations, self.n_detectors, self.measurement_dim), dtype=np.float64)
        for start in range(0, n_simulations, chunk_size):
            stop = min(start + chunk_size, n_simulations)
            for i in range(start, stop):
                x[i] = self.simulate_measurements(theta[i], rng)
            if verbose:  # pragma: no cover - progress reporting
                print(f"[gravitational_waves] simulated {stop}/{n_simulations}", flush=True)
        return {
            "theta": theta,
            "x": x,
            "joint": self.to_joint(theta, x),
            "condition_masks": self.targeted_condition_masks(),
        }

    def targeted_training_data(
        self,
        n_simulations: int = 100_000,
        *,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Per-target training data, mirroring Appendix A3.2.

        For each of the three targeted conditionals we return the observed
        ``theta``/``x`` pair together with the corresponding ``M_C`` mask; the
        condition-mask machinery of :mod:`simformer.condition_masks` will then
        only ever sample these masks during training.
        """
        data = self.make_dataset(n_simulations, rng=rng, verbose=verbose)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for target in self.targets:
            blocks = self.active_blocks_for_target(target)
            out[target] = {
                "theta": data["theta"],
                "x": data["x"][:, blocks, :],
                "joint": data["joint"],
                "condition_mask": self.condition_mask_for_target(target),
                "active_blocks": np.asarray(blocks, dtype=np.int64),
            }
        return out


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def build_task(embedding: bool = True, seed: int = 0, **kwargs: Any) -> GravitationalWavesTask:
    """Convenience constructor for :class:`GravitationalWavesTask`."""
    config = GravitationalWavesConfig(seed=int(seed), **kwargs)
    return GravitationalWavesTask(config, embedding=embedding)


def targeted_targets(names: Sequence[str] = GW_TARGETS, n_blocks: int = GW_N_DETECTORS) -> Dict[str, List[int]]:
    """Map targeted-conditional names to active detector-block indices."""
    out: Dict[str, List[int]] = {}
    for name in names:
        key = str(name).lower()
        if key in ("x1_x2", "x1x2", "both", "joint", "full"):
            out[key] = list(range(n_blocks))
        elif key in ("x1", "1", "detector1", "h1"):
            out[key] = [0]
        elif key in ("x2", "2", "detector2", "l1"):
            out[key] = [1] if n_blocks > 1 else []
        else:
            raise ValueError(f"Unknown target {name!r}")
    return out


def targeted_condition_masks(
    n_parameters: int = GW_N_PARAMETERS,
    n_detectors: int = GW_N_DETECTORS,
    targets: Sequence[str] = GW_TARGETS,
) -> Dict[str, np.ndarray]:
    """Token-level condition masks for the targeted (partial) posteriors."""
    blocks = targeted_targets(targets, n_blocks=n_detectors)
    masks: Dict[str, np.ndarray] = {}
    for name, active in blocks.items():
        theta_part = np.zeros(n_parameters, dtype=np.float32)
        data_part = np.array(
            [1.0 if k in set(active) else 0.0 for k in range(n_detectors)], dtype=np.float32
        )
        masks[name] = np.concatenate([theta_part, data_part])
    return masks


def waveform_frequency_grid(
    m1: float,
    m2: float,
    *,
    n_samples: int = GW_MEASUREMENT_DIM,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    f_min: float = DEFAULT_F_MIN,
) -> np.ndarray:
    """Instantaneous frequency of the inspiral (useful for sanity checks)."""
    _, f = inspiral_waveform(
        m1,
        m2,
        n_samples=n_samples,
        sample_rate=sample_rate,
        f_min=f_min,
        return_instantaneous_frequency=True,
    )
    return f


def matched_filter_snr(
    x_obs: np.ndarray,
    theta: np.ndarray,
    *,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    config: Optional[GravitationalWavesConfig] = None,
) -> float:
    """Coherent matched-filter SNR of a measurement given the injected parameters.

    Used only for diagnostics/plots; the network itself never sees it.
    """
    config = GravitationalWavesConfig() if config is None else config
    task = GravitationalWavesTask(config, embedding=False)
    template, _ = task.simulate_measurements(theta, np.random.default_rng(0), add_noise=False, return_clean=True)
    x_obs = np.asarray(x_obs, dtype=np.float64)
    return float(np.sum(x_obs * template) / (noise_sigma * np.linalg.norm(template)))


__all__ = [
    "SOLAR_MASS_TIME",
    "GW_MEASUREMENT_DIM",
    "GW_N_DETECTORS",
    "GW_N_PARAMETERS",
    "GW_TARGETS",
    "DEFAULT_MASS_MIN",
    "DEFAULT_MASS_MAX",
    "DEFAULT_DURATION",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_F_MIN",
    "DEFAULT_DETECTOR_RESPONSE",
    "DEFAULT_DETECTOR_DELAYS",
    "DEFAULT_DETECTOR_PHASES",
    "DEFAULT_NOISE_SIGMA",
    "GravitationalWavesConfig",
    "GravitationalWavesTask",
    "chirp_mass",
    "symmetric_mass_ratio",
    "inspiral_frequency",
    "inspiral_waveform",
    "project_to_detector",
    "sample_prior",
    "log_prior",
    "log_likelihood_gaussian",
    "waveform_frequency_grid",
    "matched_filter_snr",
    "build_task",
    "targeted_targets",
    "targeted_condition_masks",
]
