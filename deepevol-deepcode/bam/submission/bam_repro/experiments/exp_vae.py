"""Section 5.3 / Appendix E.6 experiment: black-box VI for the CIFAR-10 VAE posterior.

Deep generative model (Section 5.3 of the BaM paper)::

    z_n ~ N(0, I),      x_n | z_n ~ N(Omega(z_n, theta_hat), sigma^2 I),   sigma^2 = 0.1

The decoder ``Omega(., theta_hat)`` is pre-trained with variational EM
(factorised-Gaussian encoder, Adam with a 0 -> 1e-4 warmup over 100 steps
followed by a 1e-4 -> 1e-5 decay over 500 steps, 100 epochs, ``mc_sim=1``).

The inference problem for a held-out image ``x'`` is the posterior
``p(z' | x') propto p(z') p(x' | z')`` in ``R^latent_dim`` (``latent_dim = 256``),
whose score is available in closed form::

    grad_z log p(z' | x') = -z' + (1 / sigma^2) J_Omega(z')^T (x' - Omega(z'))

The experiment compares:

* **BaM** (this paper) with a constant inverse-regularisation ``lambda_t``
  (values taken from the addendum; small at ``B = 10``, large at ``B = 300``),
* **ADVI** (negative ELBO + ADAM),
* **Score** ADVI (score / weighted-Fisher divergence + ADAM),
* **Fisher** ADVI (Fisher divergence + ADAM),
* **GSM** (per-sample Gaussian score matching),

and reports the **reconstruction MSE** of the fitted posterior mean as a
function of the number of **gradient evaluations** (the paper's cost axis;
wallclock is explicitly out of scope).

Learning rates of the gradient-based baselines are grid searched in a short
*pilot* run (``T = 100`` iterations) before the full ``T = 1000`` iteration run;
the paper's final values (ADVI ``0.02``) are used as the default and the pilot
may be disabled.

The pre-trained decoder is cached on disk (``cache/vae_decoder.pkl``) so that the
expensive pre-training is performed only once.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# import shim: work both as a package module and as a standalone script
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly
    from ..bam.bam import BaM
    from ..bam.learning_rate import make_schedule
    from ..bam.vi_base import init_gaussian_state
    from ..baselines.advi import ADVI
    from ..baselines.fisher_advi import FisherADVI
    from ..baselines.gsm import GSM
    from ..baselines.score_advi import ScoreADVI
    from ..metrics.posteriordb_metrics import reconstruction_mse
    from ..targets.vae_target import (
        PAPER_VAE_ADVI_LR,
        PAPER_VAE_ADVI_LR_GRID,
        PAPER_VAE_BAM_LAMBDA_GRIDS,
        PAPER_VAE_BAM_LAMBDAS,
        PAPER_VAE_BATCH_SIZES,
        PAPER_VAE_GRAD_BUDGET,
        PAPER_VAE_PILOT_T,
        PAPER_VAE_T,
        PAPER_VAE_WALLCLOCK_TARGET_BATCH,
        VAE_BATCH_SIZE,
        VAE_C_HID,
        VAE_IMAGE_DIM,
        VAE_IMAGE_SHAPE,
        VAE_LATENT_DIM,
        VAE_N_EPOCHS,
        VAE_SIGMA2,
        Decoder,
        ImageData,
        VAETarget,
        amortized_reconstruction_mse,
        load_cifar10,
        load_vae,
        save_vae,
        synthetic_image_data,
        train_vae,
        vae_target_from_image,
    )
except ImportError:  # pragma: no cover - direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bam_repro.bam.bam import BaM
    from bam_repro.bam.learning_rate import make_schedule
    from bam_repro.bam.vi_base import init_gaussian_state
    from bam_repro.baselines.advi import ADVI
    from bam_repro.baselines.fisher_advi import FisherADVI
    from bam_repro.baselines.gsm import GSM
    from bam_repro.baselines.score_advi import ScoreADVI
    from bam_repro.metrics.posteriordb_metrics import reconstruction_mse
    from bam_repro.targets.vae_target import (
        PAPER_VAE_ADVI_LR,
        PAPER_VAE_ADVI_LR_GRID,
        PAPER_VAE_BAM_LAMBDA_GRIDS,
        PAPER_VAE_BAM_LAMBDAS,
        PAPER_VAE_BATCH_SIZES,
        PAPER_VAE_GRAD_BUDGET,
        PAPER_VAE_PILOT_T,
        PAPER_VAE_T,
        PAPER_VAE_WALLCLOCK_TARGET_BATCH,
        VAE_BATCH_SIZE,
        VAE_C_HID,
        VAE_IMAGE_DIM,
        VAE_IMAGE_SHAPE,
        VAE_LATENT_DIM,
        VAE_N_EPOCHS,
        VAE_SIGMA2,
        Decoder,
        ImageData,
        VAETarget,
        amortized_reconstruction_mse,
        load_cifar10,
        load_vae,
        save_vae,
        synthetic_image_data,
        train_vae,
        vae_target_from_image,
    )

__all__ = [
    "PAPER_VAE_METHODS",
    "PAPER_VAE_N_RUNS",
    "PAPER_VAE_GRAD_BUDGETS",
    "PAPER_VAE_INIT_MEAN_SCALE",
    "PAPER_VAE_SETTINGS",
    "VAEExperimentResult",
    "paper_vae_settings",
    "default_learning_rate",
    "default_lambda",
    "make_vae_problem",
    "build_learner",
    "reconstruction_curve",
    "run_vae_replicate",
    "run_vae_experiment",
    "run_vae",
    "run_pilot",
    "main",
]

# ---------------------------------------------------------------------------
# paper settings
# ---------------------------------------------------------------------------

#: Batch sizes swept in Figure 5.4 (Section 5.3).
#: (re-exported from ``targets/vae_target.py``)
PAPER_VAE_METHODS: Tuple[str, ...] = ("bam", "advi", "score", "fisher", "gsm")

#: Number of independent runs per (method, batch size) cell.  The paper does not
#: state this explicitly for the VAE experiment; a documented default of 5 is
#: used, matching the real-data (posteriorDB) protocol.
PAPER_VAE_N_RUNS: int = 5

#: Mean of the initial variational mean:  mu_0 ~ Uniform[0, 0.1]^D  (paper init).
PAPER_VAE_INIT_MEAN_SCALE: float = 0.1

#: Gradient-evaluation budgets per batch size.  The paper's x-axis is the number
#: of gradient evaluations and each panel is run to (roughly) a common budget;
#: these documented defaults give T = budget / B iterations
#: (B=10 -> 2000 iters, B=100 -> 1000 iters, B=300 -> 1000 iters).
PAPER_VAE_GRAD_BUDGETS: Dict[int, int] = {10: 20000, 100: 100000, 300: 300000}

#: Number of history snapshots kept per run (for compact JSON / plots).
PAPER_VAE_HISTORY_POINTS: int = 120

#: Default decoder pre-training settings (documented defaults for the details the
#: paper leaves unspecified; see the plan's "handling missing details").
PAPER_VAE_DECODER_CACHE: str = "vae_decoder.pkl"


def paper_vae_settings(batch_size: int = 300, **overrides: Any) -> Dict[str, Any]:
    """Return the paper-faithful configuration for one VAE batch size.

    Parameters
    ----------
    batch_size:
        Batch size ``B`` in ``{10, 100, 300}``.
    **overrides:
        Values that override the returned dictionary.
    """
    B = int(batch_size)
    settings: Dict[str, Any] = {
        "setting": "B%d" % B,
        "batch_size": B,
        "latent_dim": VAE_LATENT_DIM,
        "image_dim": VAE_IMAGE_DIM,
        "image_shape": tuple(VAE_IMAGE_SHAPE),
        "sigma2": VAE_SIGMA2,
        "n_runs": PAPER_VAE_N_RUNS,
        "pilot_T": PAPER_VAE_PILOT_T,
        "T": PAPER_VAE_T,
        "grad_budget": PAPER_VAE_GRAD_BUDGETS.get(B, PAPER_VAE_T * B),
        "methods": PAPER_VAE_METHODS,
        "bam_lambda": PAPER_VAE_BAM_LAMBDAS.get(B, None),
        "bam_lambda_grid": PAPER_VAE_BAM_LAMBDA_GRIDS.get(B, None),
        "advi_lr": PAPER_VAE_ADVI_LR,
        "advi_lr_grid": PAPER_VAE_ADVI_LR_GRID,
        "init_mean_scale": PAPER_VAE_INIT_MEAN_SCALE,
        "schedule": "constant",
        "history_points": PAPER_VAE_HISTORY_POINTS,
    }
    settings.update(overrides)
    return settings


#: Full settings dictionary for every panel of Figure 5.4.
PAPER_VAE_SETTINGS: Dict[int, Dict[str, Any]] = {
    int(B): paper_vae_settings(B) for B in PAPER_VAE_BATCH_SIZES
}


def default_learning_rate(method: str, batch_size: int = 300) -> Optional[float]:
    """Paper grid-searched ADAM learning rate for a baseline (``None`` for BaM/GSM).

    The paper reports ``0.02`` for ADVI (and for the score / Fisher variants in
    the VAE experiment); GSM has no learning rate.
    """
    method = str(method).lower()
    if method in ("bam", "gsm"):
        return None
    if method in ("advi", "score", "score_advi", "fisher", "fisher_advi"):
        return float(PAPER_VAE_ADVI_LR)
    return None


def default_lambda(method: str = "bam", batch_size: int = 300) -> Optional[float]:
    """Inverse-regularisation ``lambda`` used by BaM (addendum values)."""
    method = str(method).lower()
    if method != "bam":
        return None
    lam = PAPER_VAE_BAM_LAMBDAS.get(int(batch_size), None)
    if lam is None and PAPER_VAE_BAM_LAMBDA_GRIDS:
        grid = PAPER_VAE_BAM_LAMBDA_GRIDS.get(int(batch_size))
        if grid:
            lam = float(np.median(np.asarray(grid, dtype=float)))
    return None if lam is None else float(lam)


# ---------------------------------------------------------------------------
# problem construction (data + pre-trained decoder + target)
# ---------------------------------------------------------------------------


def ensure_vae(
    data: Optional[Any] = None,
    vae: Optional[Any] = None,
    vae_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    latent_dim: int = VAE_LATENT_DIM,
    c_hid: int = VAE_C_HID,
    image_shape: Sequence[int] = VAE_IMAGE_SHAPE,
    sigma2: float = VAE_SIGMA2,
    epochs: int = VAE_N_EPOCHS,
    batch_size: int = VAE_BATCH_SIZE,
    seed: int = 0,
    data_root: Optional[str] = None,
    subset: Optional[int] = None,
    allow_synthetic: bool = True,
    retrain: bool = False,
    verbose: bool = True,
    **train_kwargs: Any,
) -> Any:
    """Load a cached decoder or pre-train one with variational EM.

    The pre-training is expensive (100 epochs), so the resulting model is cached
    to ``cache_dir`` (default ``./cache``) and re-used afterwards.
    """
    if vae is not None:
        return vae

    if vae_path is None:
        cache_dir = cache_dir or os.path.join(os.getcwd(), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        vae_path = os.path.join(cache_dir, PAPER_VAE_DECODER_CACHE)

    if os.path.exists(vae_path) and not retrain:
        try:
            if verbose:
                print("[vae] loading cached decoder from %s" % vae_path)
            return load_vae(vae_path, c_hid=c_hid, verbose=verbose)
        except Exception as exc:  # pragma: no cover - cache corruption
            if verbose:
                print("[vae] could not load cache (%s); retraining" % exc)

    if data is None:
        data = load_cifar10(root=data_root, subset=subset, allow_synthetic=allow_synthetic, seed=seed)

    if verbose:
        print("[vae] pre-training decoder with variational EM (%d epochs) ..." % epochs)
    trained, _history = train_vae(
        data,
        latent_dim=latent_dim,
        c_hid=c_hid,
        image_shape=tuple(image_shape),
        epochs=epochs,
        batch_size=batch_size,
        sigma2=sigma2,
        mc_sim=1,
        seed=seed,
        verbose=verbose,
        **train_kwargs,
    )
    try:
        save_vae(vae_path, trained)
        if verbose:
            print("[vae] cached decoder at %s" % vae_path)
    except Exception:  # pragma: no cover - caching is best-effort
        pass
    return trained


def make_vae_problem(
    setting: Any = None,
    batch_size: int = 300,
    seed: int = 0,
    data: Optional[Any] = None,
    vae: Optional[Any] = None,
    vae_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    data_root: Optional[str] = None,
    subset: Optional[int] = None,
    image_index: int = 0,
    image: Optional[np.ndarray] = None,
    latent_dim: int = VAE_LATENT_DIM,
    sigma2: float = VAE_SIGMA2,
    mu_scale: float = PAPER_VAE_INIT_MEAN_SCALE,
    mu0: Optional[np.ndarray] = None,
    Sigma0: Optional[np.ndarray] = None,
    retrain: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Tuple[Any, np.ndarray, np.ndarray, Any, np.ndarray]:
    """Build the (target, mu0, Sigma0, vae, x) problem instance for Section 5.3.

    Returns
    -------
    target:
        :class:`~bam_repro.targets.vae_target.VAETarget` for a single held-out
        image ``x'`` (note ``target.dim == latent_dim == 256``, not the image dim).
    mu0, Sigma0:
        Paper initialisation:  ``mu_0 ~ Uniform[0, mu_scale]^D``, ``Sigma_0 = I``.
    vae:
        The pre-trained model (used for the amortised / AVI reference).
    x:
        The flattened held-out image (scaled to ``[-1, 1]``).
    """
    B = int(batch_size)
    if setting is None:
        setting = "B%d" % B

    vae = ensure_vae(
        data=data,
        vae=vae,
        vae_path=vae_path,
        cache_dir=cache_dir,
        latent_dim=latent_dim,
        image_shape=VAE_IMAGE_SHAPE,
        sigma2=sigma2,
        data_root=data_root,
        subset=subset,
        retrain=retrain,
        verbose=verbose,
        seed=seed,
    )

    if image is None:
        if data is None:
            data = load_cifar10(root=data_root, subset=subset, allow_synthetic=True, seed=seed)
        x_test = np.asarray(data.test)
        if x_test.ndim == 1:
            x_test = x_test[None, :]
        idx = int(image_index) % max(1, x_test.shape[0])
        image = x_test[idx]

    x_flat = np.asarray(image, dtype=np.float64).reshape(-1)

    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        decoder = vae  # pragma: no cover - defensive
    target = vae_target_from_image(decoder, x_flat, sigma2=sigma2, image_shape=tuple(VAE_IMAGE_SHAPE))

    dim = int(getattr(target, "dim", latent_dim))
    if mu0 is None:
        rng = np.random.default_rng(int(seed))
        mu0 = rng.uniform(0.0, float(mu_scale), size=(dim,))
        if float(mu_scale) <= 0.0:
            mu0 = np.zeros(dim)
    mu0 = np.asarray(mu0, dtype=np.float64).reshape(dim)
    if Sigma0 is None:
        Sigma0 = np.eye(dim, dtype=np.float64)
    Sigma0 = np.asarray(Sigma0, dtype=np.float64).reshape(dim, dim)
    return target, mu0, Sigma0, vae, x_flat


# ---------------------------------------------------------------------------
# learner construction
# ---------------------------------------------------------------------------


def _filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keyword arguments that ``cls`` does not accept."""
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _resolve_schedule(schedule: Any, batch_size: int, dim: int) -> Any:
    """Resolve a schedule specification to a callable ``t -> lambda_t``."""
    if callable(schedule) and not isinstance(schedule, str):
        return schedule
    if isinstance(schedule, (int, float)):
        value = float(schedule)
        return lambda t, v=value: v
    if schedule is None:
        schedule = "constant"
    try:
        return make_schedule(schedule, batch_size=batch_size, dim=dim)
    except Exception:  # pragma: no cover - defensive fallback
        table = {
            "BD": float(batch_size * dim),
            "BD/(t+1)": None,
            "BD/sqrt(t+1)": None,
            "B/(t+1)": None,
            "constant": 1.0,
        }
        if schedule in ("BD/(t+1)", "BD/sqrt(t+1)", "B/(t+1)"):
            power = 0.5 if "sqrt" in schedule else 1.0
            scale = float(batch_size * dim) if schedule.startswith("BD") else float(batch_size)
            return lambda t, s=scale, p=power: s / (t + 1.0) ** p
        return lambda t, v=table.get(schedule, 1.0): float(v)


def build_learner(
    method: str,
    target: Any,
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    batch_size: int,
    seed: int = 0,
    learning_rate: Optional[float] = None,
    lam: Optional[float] = None,
    schedule: Any = "constant",
    T: int = PAPER_VAE_T,
    history_every: int = 1,
    dim: Optional[int] = None,
    track_history: bool = True,
    verbose: bool = False,
    **extra: Any,
):
    """Instantiate the requested learner for the VAE target.

    ``method`` is one of ``"bam"``, ``"advi"``, ``"score"``, ``"fisher"``, ``"gsm"``.
    """
    method = str(method).lower()
    dim = int(dim if dim is not None else np.asarray(mu0).shape[0])
    B = int(batch_size)
    score_fn = getattr(target, "score", None) or getattr(target, "grad_log_prob")

    common: Dict[str, Any] = {
        "seed": int(seed),
        "rng": np.random.default_rng(int(seed)),
        "dtype": np.float64,
    }

    if method == "bam":
        if lam is None:
            lam = default_lambda("bam", B)
        kwargs: Dict[str, Any] = {
            "mu0": mu0,
            "Sigma0": Sigma0,
            "score_fn": score_fn,
            "batch_size": B,
            "lam": lam,
            "track_history": bool(track_history),
            "history_every": int(history_every),
            "seed": int(seed),
            "rng": np.random.default_rng(int(seed)),
            "dtype": np.float64,
        }
        if schedule not in (None, "constant"):
            kwargs["lam"] = _resolve_schedule(schedule, B, dim)
        kwargs.update(extra)
        return BaM(**_filter_kwargs(BaM, kwargs))

    if method == "gsm":
        kwargs = {
            "mu0": mu0,
            "Sigma0": Sigma0,
            "score_fn": score_fn,
            "batch_size": B,
            "track_history": bool(track_history),
            "history_every": int(history_every),
            **common,
        }
        kwargs.update(extra)
        return GSM(**_filter_kwargs(GSM, kwargs))

    cls: Any = {"advi": ADVI, "score": ScoreADVI, "fisher": FisherADVI}.get(method)
    if cls is None:
        raise ValueError("unknown method %r" % (method,))
    lr = learning_rate
    if lr is None:
        lr = default_learning_rate(method, B)
    kwargs = {
        "mu0": mu0,
        "Sigma0": Sigma0,
        "score_fn": score_fn,
        "target": target,
        "batch_size": B,
        "learning_rate": float(lr),
        "track_history": bool(track_history),
        "history_every": int(history_every),
        **common,
    }
    kwargs.update(extra)
    try:
        return cls(**_filter_kwargs(cls, kwargs))
    except TypeError:  # pragma: no cover - defensive
        kwargs.pop("dtype", None)
        return cls(**_filter_kwargs(cls, kwargs))


# ---------------------------------------------------------------------------
# metrics / history helpers
# ---------------------------------------------------------------------------


def _history_of(learner: Any) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Best-effort extraction of ``(mu_history, Sigma_history)`` from a learner."""
    mu_hist: List[np.ndarray] = []
    Sigma_hist: List[np.ndarray] = []
    for name in ("mu_history", "means", "mean_history"):
        seq = getattr(learner, name, None)
        if seq:
            mu_hist = [np.asarray(m, dtype=np.float64).reshape(-1) for m in seq]
            break
    for name in ("Sigma_history", "cov_history", "covariances"):
        seq = getattr(learner, name, None)
        if seq:
            Sigma_hist = [np.asarray(S, dtype=np.float64) for S in seq]
            break
    hist = getattr(learner, "history", None)
    if isinstance(hist, dict):
        if not mu_hist:
            seq = hist.get("mu") or hist.get("mean") or []
            mu_hist = [np.asarray(m, dtype=np.float64).reshape(-1) for m in seq]
        if not Sigma_hist:
            seq = hist.get("Sigma") or hist.get("cov") or []
            Sigma_hist = [np.asarray(S, dtype=np.float64) for S in seq]
    return mu_hist, Sigma_hist


def _grad_eval_axis(learner: Any, n_points: int, batch_size: int, history_every: int = 1) -> np.ndarray:
    """Gradient-evaluation abscissa for a history of length ``n_points``."""
    for name in ("grad_evals_history", "grad_evals_hist"):
        seq = getattr(learner, name, None)
        if seq and len(seq) >= n_points:
            return np.asarray(seq[:n_points], dtype=np.float64)
    every = max(1, int(history_every))
    offset = 0
    if getattr(learner, "mu_history", None):
        # history may include the initial state at index 0
        first_step = 0
        total = getattr(learner, "grad_evals", None)
        if total is not None and len(getattr(learner, "mu_history")) * every * batch_size > total * 1.0:
            first_step = 1
        offset = first_step
    steps = np.arange(n_points, dtype=np.float64)
    if offset:
        steps = np.maximum(steps - 1.0, 0.0)
    return (steps * float(every) + 1.0) * float(batch_size)


def _subsample(arrays: Sequence[np.ndarray], n_points: int) -> List[np.ndarray]:
    """Uniformly subsample a list of arrays to at most ``n_points`` entries."""
    n = len(arrays[0]) if arrays else 0
    if n == 0:
        return [np.asarray(a) for a in arrays]
    if n <= n_points:
        return [np.asarray(a) for a in arrays]
    idx = np.unique(np.linspace(0, n - 1, int(n_points)).round().astype(int))
    return [np.asarray(a)[idx] for a in arrays]


def reconstruction_curve(
    target: Any,
    mu_history: Sequence[np.ndarray],
    x_flat: Optional[np.ndarray] = None,
    image_shape: Sequence[int] = VAE_IMAGE_SHAPE,
    reduction: str = "mean",
) -> np.ndarray:
    """Reconstruction MSE of the decoder at each variational mean in ``mu_history``.

    ``MSE(mu) = || x' - Omega(mu) ||^2 / dim(x)`` averaged over the image; this is
    the quantity plotted in Figure 5.4.
    """
    values: List[float] = []
    for mu in mu_history:
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        mse = None
        rec_fn = getattr(target, "reconstruction_mse", None)
        if callable(rec_fn):
            try:
                mse = float(np.asarray(rec_fn(mu)).reshape(-1)[0])
            except Exception:
                mse = None
        if mse is None:
            rec = None
            rec_fn = getattr(target, "reconstruction", None)
            if callable(rec_fn):
                rec = rec_fn(mu)
            if rec is not None and x_flat is not None:
                mse = float(reconstruction_mse(x_flat, rec, reduction=reduction))
        values.append(np.nan if mse is None else float(mse))
    return np.asarray(values, dtype=np.float64)


def _final_moments(learner: Any) -> Tuple[np.ndarray, np.ndarray]:
    for name in ("mu", "mean"):
        mu = getattr(learner, name, None)
        if mu is not None:
            mu = np.asarray(mu, dtype=np.float64).reshape(-1)
            break
    else:  # pragma: no cover - defensive
        mu = np.zeros(1)
    Sigma = getattr(learner, "Sigma", None)
    if Sigma is None:
        Sigma = getattr(learner, "covariance", None)
    if Sigma is None:  # pragma: no cover - defensive
        Sigma = np.eye(mu.shape[0])
    return mu, np.asarray(Sigma, dtype=np.float64)


def _avi_reference(vae: Any, x_flat: np.ndarray, image_shape: Sequence[int] = VAE_IMAGE_SHAPE) -> float:
    """Amortised-VI (AVI) reconstruction MSE reference of Figure 5.4."""
    encoder = getattr(vae, "encoder", None)
    decoder = getattr(vae, "decoder", None)
    if encoder is None or decoder is None:
        return float("nan")
    try:
        return float(
            amortized_reconstruction_mse(decoder, encoder, np.asarray(x_flat, dtype=np.float64)[None, :], tuple(image_shape))
        )
    except Exception:  # pragma: no cover - defensive
        return float("nan")


# ---------------------------------------------------------------------------
# single replicate
# ---------------------------------------------------------------------------


def run_vae_replicate(
    setting: Any = None,
    method: str = "bam",
    batch_size: int = 300,
    seed: int = 0,
    T: Optional[int] = None,
    n_iter: Optional[int] = None,
    grad_budget: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lam: Optional[float] = None,
    schedule: Any = "constant",
    mu_scale: float = PAPER_VAE_INIT_MEAN_SCALE,
    latent_dim: int = VAE_LATENT_DIM,
    image_index: int = 0,
    history_points: int = PAPER_VAE_HISTORY_POINTS,
    target: Optional[Any] = None,
    vae: Optional[Any] = None,
    data: Optional[Any] = None,
    x_flat: Optional[np.ndarray] = None,
    mu0: Optional[np.ndarray] = None,
    Sigma0: Optional[np.ndarray] = None,
    return_learner: bool = False,
    verbose: bool = False,
    **learner_kwargs: Any,
) -> Dict[str, Any]:
    """Run a single (method, batch size, seed) replicate for the VAE target."""
    B = int(batch_size)
    if T is None:
        T = PAPER_VAE_T if n_iter is None else int(n_iter)
    T = int(T)
    if grad_budget is not None and int(grad_budget) > 0:
        T = max(1, int(int(grad_budget) // max(1, B)))

    if target is None:
        target, mu0, Sigma0, vae, x_flat = make_vae_problem(
            setting=setting,
            batch_size=B,
            seed=seed,
            data=data,
            vae=vae,
            image_index=image_index,
            latent_dim=latent_dim,
            mu_scale=mu_scale,
            mu0=mu0,
            Sigma0=Sigma0,
            verbose=verbose,
        )
    else:
        if mu0 is None or Sigma0 is None:
            dim = int(getattr(target, "dim", latent_dim))
            rng = np.random.default_rng(int(seed))
            mu0 = rng.uniform(0.0, float(mu_scale), size=(dim,)) if mu0 is None else np.asarray(mu0).reshape(-1)
            Sigma0 = np.eye(dim) if Sigma0 is None else np.asarray(Sigma0)
        if x_flat is None:
            x_flat = getattr(target, "x", None)
            if x_flat is not None:
                x_flat = np.asarray(x_flat, dtype=np.float64).reshape(-1)

    dim = int(getattr(target, "dim", latent_dim))
    history_points = int(max(2, history_points))
    history_every = max(1, int(T) // history_points)

    learner = build_learner(
        method,
        target,
        mu0,
        Sigma0,
        batch_size=B,
        seed=seed,
        learning_rate=learning_rate,
        lam=lam,
        schedule=schedule,
        T=T,
        history_every=history_every,
        dim=dim,
        verbose=verbose,
        **learner_kwargs,
    )

    t0 = time.time()
    try:
        try:
            learner.run(T)
        except AttributeError:  # pragma: no cover - functional learners
            learner.fit(T)
    except Exception as exc:  # pragma: no cover - keep sweeps alive
        if verbose:
            print("[vae] %s B=%d run %d failed: %s" % (method, B, seed, exc))
        raise
    wallclock = time.time() - t0

    mu_hist, _Sigma_hist = _history_of(learner)
    if not mu_hist:
        mu_hist = [np.asarray(learner.mu, dtype=np.float64).reshape(-1)]
    rec_curve = reconstruction_curve(target, mu_hist, x_flat=x_flat)
    grad_evals = _grad_eval_axis(learner, len(mu_hist), B, history_every)
    n = min(len(grad_evals), len(rec_curve))
    grad_evals, rec_curve = grad_evals[:n], rec_curve[:n]

    mu, Sigma = _final_moments(learner)
    final_mse = float(reconstruction_curve(target, [mu], x_flat=x_flat)[0])
    finite = np.isfinite(rec_curve)
    best_mse = float(np.nanmin(rec_curve)) if finite.any() else float("nan")
    diverged = bool(
        (not finite.any())
        or (np.nanmax(rec_curve) > 1e6 if finite.any() else True)
        or (not np.all(np.isfinite(mu)))
    )

    record: Dict[str, Any] = {
        "setting": str(setting if setting is not None else "B%d" % B),
        "batch_size": B,
        "method": str(method),
        "run": int(seed),
        "seed": int(seed),
        "image_index": int(image_index),
        "grad_evals": np.asarray(grad_evals, dtype=np.float64),
        "reconstruction_mse": np.asarray(rec_curve, dtype=np.float64),
        "final_reconstruction_mse": final_mse,
        "best_reconstruction_mse": best_mse,
        "avi_reconstruction_mse": _avi_reference(vae, x_flat) if vae is not None and x_flat is not None else float("nan"),
        "iterations": int(T),
        "grad_evals_total": float(B * T),
        "wallclock_seconds": float(wallclock),
        "lam": float(lam) if (lam is not None and np.isscalar(lam)) else (default_lambda(method, B) if method == "bam" else None),
        "learning_rate": float(learning_rate) if learning_rate is not None else default_learning_rate(method, B),
        "diverged": diverged,
        "mu": np.asarray(mu, dtype=np.float64),
        "Sigma_diag": np.diag(np.asarray(Sigma, dtype=np.float64)) if np.asarray(Sigma).ndim == 2 else None,
    }
    if return_learner:
        record["_learner"] = learner
        record["_target"] = target
    return record


# ---------------------------------------------------------------------------
# pilot: learning-rate / lambda selection
# ---------------------------------------------------------------------------


def run_pilot(
    method: str,
    batch_size: int = 300,
    T: int = PAPER_VAE_PILOT_T,
    learning_rates: Optional[Sequence[float]] = None,
    lambdas: Optional[Sequence[float]] = None,
    seeds: Sequence[int] = (0,),
    target: Optional[Any] = None,
    vae: Optional[Any] = None,
    data: Optional[Any] = None,
    image_index: int = 0,
    latent_dim: int = VAE_LATENT_DIM,
    mu_scale: float = PAPER_VAE_INIT_MEAN_SCALE,
    verbose: bool = True,
    **learner_kwargs: Any,
) -> Dict[str, Any]:
    """Short pilot run selecting the learning rate (or ``lambda``) with the
    lowest final reconstruction MSE, as described for the VAE experiment.

    Returns a dict with keys ``method``, ``T``, ``best``, ``candidates`` and
    ``scores``.
    """
    method = str(method).lower()
    B = int(batch_size)
    if target is None:
        target, mu0, Sigma0, vae, x_flat = make_vae_problem(
            setting="B%d" % B,
            batch_size=B,
            seed=0,
            data=data,
            vae=vae,
            image_index=image_index,
            latent_dim=latent_dim,
            mu_scale=mu_scale,
            verbose=verbose,
        )
    else:
        dim = int(getattr(target, "dim", latent_dim))
        mu0 = np.random.default_rng(0).uniform(0.0, float(mu_scale), size=(dim,))
        Sigma0 = np.eye(dim)

    if method == "bam":
        candidates = list(lambdas if lambdas is not None else (PAPER_VAE_BAM_LAMBDA_GRIDS.get(B) or [default_lambda("bam", B)]))
    else:
        candidates = list(learning_rates if learning_rates is not None else PAPER_VAE_ADVI_LR_GRID)
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        candidates = [None]

    scores: List[float] = []
    for cand in candidates:
        vals: List[float] = []
        for s in seeds:
            kwargs: Dict[str, Any] = dict(learner_kwargs)
            kwargs["T"] = int(T)
            if method == "bam":
                kwargs["lam"] = cand
            else:
                kwargs["learning_rate"] = cand
            try:
                rec = run_vae_replicate(
                    setting="B%d" % B,
                    method=method,
                    batch_size=B,
                    seed=int(s) + 4242,
                    target=target,
                    vae=vae,
                    x_flat=x_flat,
                    mu0=mu0,
                    Sigma0=Sigma0,
                    image_index=image_index,
                    verbose=False,
                    **kwargs,
                )
                vals.append(float(rec["final_reconstruction_mse"]))
            except Exception:  # pragma: no cover - a candidate may diverge
                vals.append(float("inf"))
        score = float(np.mean(vals)) if vals else float("inf")
        scores.append(score)
        if verbose:
            label = "lambda" if method == "bam" else "lr"
            print("[pilot] %s B=%d %s=%g -> final MSE %.6e" % (method, B, label, cand, score))

    best_idx = int(np.argmin(scores)) if scores else 0
    return {
        "method": method,
        "batch_size": B,
        "T": int(T),
        "candidates": candidates,
        "scores": scores,
        "best": candidates[best_idx],
    }


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class VAEExperimentResult:
    """Aggregated results of the Section 5.3 (Figure 5.4) VAE experiment."""

    settings: List[Dict[str, Any]] = field(default_factory=list)
    methods: Tuple[str, ...] = PAPER_VAE_METHODS
    n_runs: int = PAPER_VAE_N_RUNS
    config: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    curves: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    # -- aggregation -------------------------------------------------------
    def aggregate(self, grid_points: int = 60) -> "VAEExperimentResult":
        """Average the per-run reconstruction curves onto a common grad-eval grid."""
        curves: Dict[str, Any] = {}
        summary: Dict[str, Any] = {}
        by_setting: Dict[str, List[Dict[str, Any]]] = {}
        for rec in self.records:
            by_setting.setdefault(str(rec["setting"]), []).append(rec)

        for setting, recs in by_setting.items():
            methods = sorted({str(r["method"]) for r in recs})
            lo = max(1.0, min(float(np.min(r["grad_evals"])) for r in recs if len(r["grad_evals"])))
            hi = max(float(np.max(r["grad_evals"])) for r in recs if len(r["grad_evals"]))
            grid = np.logspace(math.log10(lo), math.log10(max(hi, lo * 10.0)), int(grid_points))
            curves[setting] = {"grid": grid, "methods": {}}
            summary[setting] = {}
            for method in methods:
                runs = [r for r in recs if str(r["method"]) == method]
                stack = []
                for r in runs:
                    x = np.asarray(r["grad_evals"], dtype=np.float64)
                    y = np.asarray(r["reconstruction_mse"], dtype=np.float64)
                    if x.size == 0:
                        continue
                    order = np.argsort(x)
                    x, y = x[order], y[order]
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 2:
                        continue
                    stack.append(np.interp(grid, x[mask], y[mask], left=np.nan, right=np.nan))
                if not stack:
                    continue
                mat = np.vstack(stack)
                mean = np.nanmean(mat, axis=0)
                with np.errstate(invalid="ignore"):
                    stderr = np.nanstd(mat, axis=0, ddof=1) / math.sqrt(max(1, mat.shape[0]))
                mask = np.isfinite(mean)
                curves[setting]["methods"][method] = {
                    "mean": mean,
                    "stderr": stderr if np.ndim(stderr) else np.zeros_like(mean),
                    "n_runs": int(mat.shape[0]),
                    "runs": mat,
                }
                finals = [float(r["final_reconstruction_mse"]) for r in runs]
                finals = [v for v in finals if np.isfinite(v)]
                summary[setting][method] = {
                    "n_runs": len(runs),
                    "final_reconstruction_mse_mean": float(np.mean(finals)) if finals else float("nan"),
                    "final_reconstruction_mse_std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
                    "final_reconstruction_mse_stderr": float(np.std(finals, ddof=1) / math.sqrt(len(finals)))
                    if len(finals) > 1
                    else 0.0,
                    "best_reconstruction_mse_mean": float(
                        np.mean([float(r["best_reconstruction_mse"]) for r in runs if np.isfinite(r["best_reconstruction_mse"])])
                    )
                    if any(np.isfinite(r["best_reconstruction_mse"]) for r in runs)
                    else float("nan"),
                    "avi_reconstruction_mse": float(
                        np.nanmean([float(r.get("avi_reconstruction_mse", float("nan"))) for r in runs])
                    ),
                    "mean_wallclock_seconds": float(np.mean([float(r["wallclock_seconds"]) for r in runs])),
                    "n_diverged": int(sum(bool(r.get("diverged")) for r in runs)),
                    "mask_first": int(np.argmax(mask)) if mask.any() else -1,
                }
            curves[setting]["grid"] = grid

        # gradient evaluations required to reach a fixed reconstruction quality
        targets = [0.01, 0.005, 0.002]
        for setting, per_method in summary.items():
            for method, stats in per_method.items():
                entry = curves.get(setting, {}).get("methods", {}).get(method)
                if not entry:
                    continue
                mean = np.asarray(entry["mean"], dtype=np.float64)
                grad = np.asarray(curves[setting]["grid"], dtype=np.float64)
                for thr in targets:
                    idx = np.where(np.isfinite(mean) & (mean <= thr))[0]
                    stats["grad_evals_to_%.4g" % thr] = float(grad[idx[0]]) if idx.size else float("nan")

        self.curves = curves
        self.summary = summary
        return self

    # -- reporting ---------------------------------------------------------
    def table(self) -> str:
        lines = []
        header = "%-8s %-8s %14s %14s %12s %10s" % (
            "setting", "method", "final MSE", "best MSE", "AVI MSE", "diverged",
        )
        lines.append(header)
        lines.append("-" * len(header))
        for setting in sorted(self.summary):
            for method in sorted(self.summary[setting]):
                s = self.summary[setting][method]
                lines.append(
                    "%-8s %-8s %14.6e %14.6e %12.6e %10d"
                    % (
                        setting,
                        method,
                        s["final_reconstruction_mse_mean"],
                        s["best_reconstruction_mse_mean"],
                        s["avi_reconstruction_mse"],
                        s["n_diverged"],
                    )
                )
        return "\n".join(lines)

    def to_dict(self, include_runs: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "settings": self.settings,
            "methods": list(self.methods),
            "n_runs": int(self.n_runs),
            "config": self.config,
            "summary": self.summary,
            "curves": {},
        }
        for setting, entry in self.curves.items():
            out["curves"][setting] = {"grid": np.asarray(entry["grid"]).tolist(), "methods": {}}
            for method, data in entry["methods"].items():
                out["curves"][setting]["methods"][method] = {
                    "mean": np.asarray(data["mean"]).tolist(),
                    "stderr": np.asarray(data["stderr"]).tolist(),
                    "n_runs": int(data["n_runs"]),
                }
                if include_runs:
                    out["curves"][setting]["methods"][method]["runs"] = np.asarray(data["runs"]).tolist()
        if include_runs:
            runs_out = []
            for rec in self.records:
                r = {k: v for k, v in rec.items() if not k.startswith("_")}
                for key in ("grad_evals", "reconstruction_mse"):
                    if isinstance(r.get(key), np.ndarray):
                        r[key] = r[key].tolist()
                if isinstance(r.get("mu"), np.ndarray):
                    r["mu"] = r["mu"].tolist()
                if isinstance(r.get("Sigma_diag"), np.ndarray):
                    r["Sigma_diag"] = r["Sigma_diag"].tolist()
                runs_out.append(r)
            out["records"] = runs_out
        return out

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(include_runs=True), fh)
        return path

    @classmethod
    def load(cls, path: str) -> "VAEExperimentResult":
        with open(path, "r") as fh:
            payload = json.load(fh)
        records = []
        for rec in payload.get("records", []):
            rec = dict(rec)
            for key in ("grad_evals", "reconstruction_mse", "mu", "Sigma_diag"):
                if rec.get(key) is not None:
                    rec[key] = np.asarray(rec[key], dtype=np.float64)
            records.append(rec)
        result = cls(
            settings=payload.get("settings", []),
            methods=tuple(payload.get("methods", PAPER_VAE_METHODS)),
            n_runs=int(payload.get("n_runs", PAPER_VAE_N_RUNS)),
            config=payload.get("config", {}),
            records=records,
        )
        result.summary = payload.get("summary", {})
        result.curves = payload.get("curves", {})
        return result

    # -- figures -----------------------------------------------------------
    def figure(
        self,
        outdir: Optional[str] = None,
        metric: str = "reconstruction_mse",
        show: bool = False,
        xlabel: str = "gradient evaluations",
        ylabel: Optional[str] = None,
    ) -> List[str]:
        """Reproduce Figure 5.4: reconstruction MSE vs gradient evaluations."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - matplotlib missing
            return []

        outdir = outdir or os.path.join(os.getcwd(), "results")
        os.makedirs(outdir, exist_ok=True)
        ylabel = ylabel or "reconstruction MSE"

        settings = sorted(self.curves, key=lambda s: int(str(s).lstrip("B") or 0))
        if not settings:
            return []
        ncols = len(settings)
        fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 3.8), squeeze=False)
        saved: List[str] = []
        for ax, setting in zip(axes[0], settings):
            entry = self.curves[setting]
            grid = np.asarray(entry["grid"])
            for method, data in sorted(entry["methods"].items()):
                mean = np.asarray(data["mean"])
                ax.plot(grid, mean, label=method)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_title(str(setting))
            ax.grid(True, which="both", alpha=0.3)
        axes[0][0].set_ylabel(ylabel)
        axes[0][-1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = os.path.join(outdir, "fig_vae_%s.png" % metric)
        fig.savefig(path, dpi=150)
        if show:  # pragma: no cover - interactive
            plt.show()
        plt.close(fig)
        saved.append(path)
        return saved


# ---------------------------------------------------------------------------
# full sweep
# ---------------------------------------------------------------------------


def _summary_sort_key(item: Tuple[str, Any]) -> Tuple[int, str]:
    setting, _ = item
    digits = "".join(ch for ch in str(setting) if ch.isdigit())
    return (int(digits) if digits else 0, str(setting))


def run_vae_experiment(
    settings: Optional[Sequence[Any]] = None,
    n_runs: int = PAPER_VAE_N_RUNS,
    methods: Sequence[str] = PAPER_VAE_METHODS,
    batch_sizes: Sequence[int] = PAPER_VAE_BATCH_SIZES,
    T: int = PAPER_VAE_T,
    pilot_T: int = PAPER_VAE_PILOT_T,
    pilot: bool = True,
    pilot_seeds: Sequence[int] = (0,),
    grad_budget: Optional[int] = None,
    budget_scale: float = 1.0,
    mu_scale: float = PAPER_VAE_INIT_MEAN_SCALE,
    latent_dim: int = VAE_LATENT_DIM,
    image_index: int = 0,
    data: Optional[Any] = None,
    vae: Optional[Any] = None,
    vae_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    data_root: Optional[str] = None,
    subset: Optional[int] = None,
    retrain: bool = False,
    seed: int = 0,
    history_points: int = PAPER_VAE_HISTORY_POINTS,
    learning_rates: Optional[Dict[str, float]] = None,
    lambdas: Optional[Dict[int, float]] = None,
    outdir: Optional[str] = None,
    save: bool = True,
    figures: bool = True,
    verbose: bool = True,
    **learner_kwargs: Any,
) -> VAEExperimentResult:
    """Run the full Section 5.3 sweep: methods x batch sizes x ``n_runs``.

    The decoder is pre-trained (or loaded from cache) **once** and shared by every
    method/run.  For each batch size a single held-out image is used, so all
    methods solve exactly the same target posterior.
    """
    batch_sizes = tuple(int(b) for b in batch_sizes)
    methods = tuple(str(m).lower() for m in methods)
    if settings is not None:
        batch_sizes = tuple(
            int(s["batch_size"]) if isinstance(s, dict) else int(str(s).lstrip("B") or 0) for s in settings
        )
    batch_sizes = tuple(b for b in batch_sizes if b > 0)
    if not batch_sizes:
        batch_sizes = tuple(int(b) for b in PAPER_VAE_BATCH_SIZES)

    # --- data + shared pre-trained decoder --------------------------------
    if data is None:
        if verbose:
            print("[vae] loading CIFAR-10 (synthetic fallback if unavailable) ...")
        data = load_cifar10(root=data_root, subset=subset, allow_synthetic=True, seed=seed)
    vae = ensure_vae(
        data=data,
        vae=vae,
        vae_path=vae_path,
        cache_dir=cache_dir,
        latent_dim=latent_dim,
        data_root=data_root,
        subset=subset,
        retrain=retrain,
        verbose=verbose,
        seed=seed,
    )

    cfg: Dict[str, Any] = {
        "T": int(T),
        "pilot_T": int(pilot_T),
        "pilot": bool(pilot),
        "n_runs": int(n_runs),
        "batch_sizes": list(batch_sizes),
        "methods": list(methods),
        "latent_dim": int(latent_dim),
        "image_index": int(image_index),
        "budget_scale": float(budget_scale),
        "grad_budget": grad_budget,
        "sigma2": float(VAE_SIGMA2),
        "seed": int(seed),
        "cost_axis": "gradient evaluations",
        "wallclock_out_of_scope": True,
    }

    result = VAEExperimentResult(
        settings=[paper_vae_settings(B) for B in batch_sizes],
        methods=methods,
        n_runs=int(n_runs),
        config=cfg,
    )

    for B in batch_sizes:
        if verbose:
            print("=" * 70)
            print("[vae] batch size B=%d" % B)
        target, mu0, Sigma0, vae, x_flat = make_vae_problem(
            setting="B%d" % B,
            batch_size=B,
            seed=seed,
            data=data,
            vae=vae,
            image_index=image_index,
            latent_dim=latent_dim,
            mu_scale=mu_scale,
            verbose=verbose,
        )
        budget = grad_budget
        if budget is None:
            budget = int(PAPER_VAE_GRAD_BUDGETS.get(B, T * B))
        budget = int(max(B, budget * float(budget_scale)))

        # --- pilot: select learning rate / lambda per method ---------------
        selected_lr: Dict[str, float] = dict(learning_rates or {})
        selected_lam: Dict[str, float] = {}
        if lambdas:
            selected_lam.update({str(k): float(v) for k, v in lambdas.items()})
        if pilot:
            for method in methods:
                if method in ("score", "fisher") and method in selected_lr:
                    continue
                if method == "bam" and str(B) in selected_lam:
                    continue
                if method in ("gsm",):
                    continue
                try:
                    pilot_out = run_pilot(
                        method,
                        batch_size=B,
                        T=pilot_T,
                        seeds=pilot_seeds,
                        target=target,
                        vae=vae,
                        x_flat=x_flat,
                        mu0=mu0,
                        Sigma0=Sigma0,
                        image_index=image_index,
                        latent_dim=latent_dim,
                        mu_scale=mu_scale,
                        verbose=verbose,
                        **learner_kwargs,
                    )
                    best = pilot_out["best"]
                    if method == "bam":
                        if best is not None:
                            selected_lam[str(B)] = float(best)
                    elif best is not None:
                        selected_lr[method] = float(best)
                except Exception as exc:  # pragma: no cover - pilot is best effort
                    if verbose:
                        print("[vae] pilot for %s failed (%s); using paper default" % (method, exc))

        # --- main runs -----------------------------------------------------
        for method in methods:
            lr = selected_lr.get(method, default_learning_rate(method, B))
            lam = selected_lam.get(str(B), default_lambda(method, B))
            for run in range(int(n_runs)):
                run_seed = int(seed) + 1000 * int(run) + 7 * int(B)
                t0 = time.time()
                try:
                    rec = run_vae_replicate(
                        setting="B%d" % B,
                        method=method,
                        batch_size=B,
                        seed=run_seed,
                        T=T,
                        grad_budget=budget,
                        learning_rate=lr,
                        lam=lam,
                        mu_scale=mu_scale,
                        latent_dim=latent_dim,
                        image_index=image_index,
                        history_points=history_points,
                        target=target,
                        vae=vae,
                        x_flat=x_flat,
                        mu0=mu0,
                        Sigma0=Sigma0,
                        verbose=False,
                        **learner_kwargs,
                    )
                    rec["run"] = int(run)
                    rec["seed"] = int(run_seed)
                    result.records.append(rec)
                    if verbose:
                        print(
                            "[vae] B=%-4d %-6s run %d/%d  MSE=%.4e  (%.1fs)"
                            % (
                                B,
                                method,
                                run + 1,
                                n_runs,
                                rec["final_reconstruction_mse"],
                                time.time() - t0,
                            )
                        )
                except Exception as exc:  # pragma: no cover - keep sweep alive
                    if verbose:
                        print("[vae] B=%d %s run %d failed: %s" % (B, method, run, exc))

    result.aggregate()

    if save:
        outdir = outdir or os.path.join(os.getcwd(), "results")
        path = os.path.join(outdir, "vae_results.json")
        result.save(path)
        if verbose:
            print("[vae] saved results to %s" % path)
    if figures:
        paths = result.figure(outdir=outdir)
        if verbose and paths:
            print("[vae] wrote figure %s" % paths[0])
    if verbose:
        print(result.table())
    return result


def run_vae(quick: bool = False, **kwargs: Any) -> VAEExperimentResult:
    """Convenience wrapper; ``quick=True`` runs a tiny smoke test."""
    if quick:
        kwargs.setdefault("batch_sizes", (10,))
        kwargs.setdefault("n_runs", 1)
        kwargs.setdefault("T", 20)
        kwargs.setdefault("pilot", False)
        kwargs.setdefault("budget_scale", 0.02)
        kwargs.setdefault("latent_dim", kwargs.pop("latent_dim", VAE_LATENT_DIM))
        kwargs.setdefault("figures", False)
        kwargs.setdefault("save", False)
        kwargs.setdefault("verbose", True)
    return run_vae_experiment(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Section 5.3 VAE experiment (Figure 5.4)")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=list(PAPER_VAE_BATCH_SIZES))
    parser.add_argument("--methods", type=str, nargs="*", default=list(PAPER_VAE_METHODS))
    parser.add_argument("--runs", type=int, default=PAPER_VAE_N_RUNS)
    parser.add_argument("--T", type=int, default=PAPER_VAE_T)
    parser.add_argument("--pilot-T", type=int, default=PAPER_VAE_PILOT_T)
    parser.add_argument("--no-pilot", action="store_true")
    parser.add_argument("--grad-budget", type=int, default=None)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=VAE_LATENT_DIM)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--history-points", type=int, default=PAPER_VAE_HISTORY_POINTS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        run_vae(quick=True, data_root=args.data_root, cache_dir=args.cache_dir)
        return 0

    result = run_vae_experiment(
        batch_sizes=tuple(args.batch_sizes),
        methods=tuple(args.methods),
        n_runs=int(args.runs),
        T=int(args.T),
        pilot_T=int(args.pilot_T),
        pilot=not args.no_pilot,
        grad_budget=args.grad_budget,
        budget_scale=float(args.budget_scale),
        latent_dim=int(args.latent_dim),
        image_index=int(args.image_index),
        data_root=args.data_root,
        subset=args.subset,
        cache_dir=args.cache_dir,
        vae_path=args.vae_path,
        retrain=bool(args.retrain),
        seed=int(args.seed),
        history_points=int(args.history_points),
        outdir=args.outdir,
        figures=not args.no_figures,
        save=not args.no_save,
        verbose=True,
    )
    print(result.table())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
