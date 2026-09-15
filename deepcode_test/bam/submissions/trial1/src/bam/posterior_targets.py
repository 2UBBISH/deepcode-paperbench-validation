"""Posteriordb / BridgeStan posterior targets.

This module wraps hierarchical Bayesian posterior densities from
``posteriordb`` through ``bridgestan`` so the Batch-and-Match algorithm and
the ADVI / Score / Fisher / GSM baselines can evaluate unnormalized
log-posterior values and their gradients (scores).

The three models used in Section 5.2 of the paper are:

* ``eight-schools-centered``  (D = 10)
* ``ark``                     (D = 7)
* ``gp-pois-regr``            (D = 13)

Exact posteriordb model naming varies slightly between releases, so several
common name aliases are accepted.  If ``posteriordb`` or ``bridgestan`` is
not installed, importing this module still succeeds; calling the loaders
then raises an informative :class:`RuntimeError`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

import jax
import jax.numpy as jnp

try:  # pragma: no cover - availability check
    import bridgestan  # type: ignore

    _HAVE_BRIDGESTAN = True
except Exception:  # noqa: BLE001 - optional dependency
    bridgestan = None  # type: ignore
    _HAVE_BRIDGESTAN = False

try:  # pragma: no cover - availability check
    import posteriordb  # type: ignore

    _HAVE_POSTERIORDB = True
except Exception:  # noqa: BLE001 - optional dependency
    posteriordb = None  # type: ignore
    _HAVE_POSTERIORDB = False


# ---------------------------------------------------------------------------
# Registry of Section 5.2 posteriors.  ``stan_name`` is the human-readable
# model name, ``dim`` the posterior dimension, and ``pdb_names`` are the
# aliases accepted by ``posteriordb``.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PosteriorSpec:
    stan_name: str
    dim: int
    pdb_names: Tuple[str, ...]
    data_name: Optional[str] = None


POSTERIOR_SPECS: dict[str, PosteriorSpec] = {
    "ark": PosteriorSpec(
        stan_name="ark",
        dim=7,
        pdb_names=("ark", "ark-arK"),
    ),
    "gp-pois-regr": PosteriorSpec(
        stan_name="gp-pois-regr",
        dim=13,
        pdb_names=("gp-pois-regr", "gp_pois_regr", "gp-pois-regr-gp_pois_regr"),
    ),
    "eight-schools-centered": PosteriorSpec(
        stan_name="eight_schools-eight_schools_centered",
        dim=10,
        pdb_names=(
            "eight_schools-eight_schools_centered",
            "eight_schools_centered",
            "eight-schools-centered",
            "eight_schools-eight_schools_centered-eight_schools_centered",
        ),
    ),
}

# Human aliases -> canonical registry key.
_MODEL_ALIASES: dict[str, str] = {}
for _key, _spec in POSTERIOR_SPECS.items():
    _MODEL_ALIASES[_key] = _key
    for _alias in _spec.pdb_names:
        _MODEL_ALIASES[_alias] = _key
    _MODEL_ALIASES[_spec.stan_name] = _key


def canonical_model_name(name: str) -> str:
    """Return the canonical registry key for *name*.

    Raises ``ValueError`` if the name is unknown.
    """
    if name in _MODEL_ALIASES:
        return _MODEL_ALIASES[name]
    raise ValueError(
        f"Unknown posterior model {name!r}. Known models: "
        f"{sorted(POSTERIOR_SPECS)}"
    )


# ---------------------------------------------------------------------------
# PosteriorTarget
# ---------------------------------------------------------------------------
class PosteriorTarget:
    """Posterior target with BridgeStan-backed log-density and score.

    Parameters
    ----------
    name:
        Canonical model name (one of the keys of :data:`POSTERIOR_SPECS`).
    dim:
        Dimension of the unconstrained posterior parameter vector.
    log_density_fn:
        Callable ``(theta: np.ndarray | jnp.ndarray) -> float`` evaluating the
        unnormalized log posterior at a single point.
    log_density_grad_fn:
        Optional callable returning ``(log_density, gradient)`` as a pair.
        When supplied, it is used for ``score``; otherwise the gradient is
        approximated through :func:`jax.grad` (requires a JAX-traceable
        density).
    reference_draws:
        Optional ``(N, D)`` array of HMC reference draws used for posterior
        mean / SD error evaluation and for drawing posterior samples.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        log_density_fn: Callable[[jnp.ndarray], float],
        log_density_grad_fn: Optional[
            Callable[[jnp.ndarray], Tuple[float, jnp.ndarray]]
        ] = None,
        reference_draws: Optional[jnp.ndarray] = None,
    ) -> None:
        self.name = name
        self.dim = int(dim)
        self._log_density = log_density_fn
        self._log_density_grad = log_density_grad_fn
        self._reference_draws = reference_draws

    # -- public evaluation API ------------------------------------------------
    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate unnormalized log posterior for a batch ``(B, D)``."""
        z = jnp.asarray(z)
        if z.ndim == 1:
            z = z[None, :]
        vals = [self._log_density(z_i) for z_i in z]
        out = jnp.asarray(vals, dtype=jnp.result_type(float))
        return out.reshape(z.shape[:-1])

    def score(self, z: jnp.ndarray) -> jnp.ndarray:
        """Evaluate ``grad_z log p(z)`` for a batch ``(B, D)``."""
        z = jnp.asarray(z)
        single = z.ndim == 1
        if single:
            z = z[None, :]
        if self._log_density_grad is not None:
            grads = [self._log_density_grad(z_i)[1] for z_i in z]
        else:
            _grad_fn = jax.grad(lambda x: self._log_density(x))
            grads = [_grad_fn(z_i) for z_i in z]
        out = jnp.asarray(grads, dtype=jnp.result_type(float))
        return out[0] if single else out

    def log_prob_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a batched log-density callable (convenience for baselines)."""
        return self.log_prob

    def score_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return a batched score callable consumed by BaM/baselines."""
        return self.score

    def sample(self, key: jax.Array, num_samples: int) -> jnp.ndarray:
        """Return posterior samples.

        Prefers loaded HMC reference draws (sampled without replacement up to
        ``num_samples``).  If fewer draws are available, draws are recycled.
        If no reference draws are present, raises ``RuntimeError``.
        """
        del key  # reference draws are fixed; key kept for API symmetry
        draws = self.reference_draws()
        if draws is None or draws.shape[0] == 0:
            raise RuntimeError(
                f"No reference draws loaded for posterior {self.name!r}; "
                "pass reference_draws= when constructing PosteriorTarget."
            )
        n = int(num_samples)
        total = draws.shape[0]
        if n <= total:
            # Deterministic first-n slice avoids repeatedly sampling the same
            # Monte-Carlo set during diagnostics.
            return jnp.asarray(draws[:n])
        repeats = n // total
        remainder = n % total
        parts = [draws] * repeats
        if remainder:
            parts.append(draws[:remainder])
        return jnp.concatenate([jnp.asarray(p) for p in parts], axis=0)

    def reference_draws(self) -> Optional[jnp.ndarray]:
        """Return loaded HMC reference draws or ``None``."""
        return self._reference_draws

    # -- dunders --------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PosteriorTarget(name={self.name!r}, dim={self.dim}, "
            f"reference_draws={None if self._reference_draws is None else self._reference_draws.shape})"
        )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _require_bridgestan() -> None:
    if not _HAVE_BRIDGESTAN:
        raise RuntimeError(
            "bridgestan is required for posterior targets but is not "
            "installed. Install it with `pip install bridgestan`."
        )


def _require_posteriordb() -> None:
    if not _HAVE_POSTERIORDB:
        raise RuntimeError(
            "posteriordb is required to automatically locate Stan models and "
            "reference draws but is not installed. Install it with "
            "`pip install posteriordb`."
        )


def _as_numpy(theta: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
    return np.asarray(jnp.asarray(theta))


def _find_posterior_by_aliases(pdb: Any, name: str) -> Any:
    """Locate a posteriordb posterior object accepting several aliases."""
    canonical = canonical_model_name(name)
    spec = POSTERIOR_SPECS[canonical]
    last_err: Optional[Exception] = None
    for alias in (name,) + spec.pdb_names:
        try:
            return pdb.posterior(alias)
        except Exception as exc:  # noqa: BLE001 - try next alias
            last_err = exc
    # Fall back to iterating the database's posteriors.
    try:
        for p in pdb.posteriors():
            pname = getattr(p, "name", "")
            if canonical in pname or pname in spec.pdb_names:
                return p
    except Exception:  # noqa: BLE001 - ignore and re-raise original
        pass
    raise ValueError(
        f"Could not find posteriordb posterior for {name!r}. Last error: {last_err}"
    )


def _model_code(posterior: Any) -> str:
    """Extract Stan model code from a posteriordb posterior object."""
    model = getattr(posterior, "model", None)
    if model is None:
        raise ValueError("posteriordb posterior has no `.model` attribute")
    for attr in ("code", "stan_code", "code_file"):
        val = getattr(model, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:  # noqa: BLE001
                continue
        if isinstance(val, str):
            return val
    raise ValueError("Could not extract Stan model code from posterior model")


def _model_data(posterior: Any) -> Optional[str]:
    """Extract Stan JSON data for a posteriordb posterior object."""
    data = getattr(posterior, "data", None)
    if data is None:
        return None
    for attr in ("data_file", "values", "file"):
        val = getattr(data, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:  # noqa: BLE001
                continue
        if isinstance(val, (str, dict)):
            return val
    return None


def _write_stan_model(code: str, directory: str, name: str) -> str:
    """Write Stan code to a file and return its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.stan")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _load_reference_draws(draws: Any) -> Optional[np.ndarray]:
    """Normalize posteriordb reference draws to an ``(N, D)`` ndarray.

    Supports ``numpy`` arrays, ``(N, D)`` array-likes, dicts whose values are
    parameter names -> 1-D draw arrays, and file paths pointing at ``.csv``,
    ``.npz``, or whitespace-delimited text.
    """
    if draws is None:
        return None

    # Callable getter.
    if callable(draws) and not isinstance(draws, (np.ndarray, list, dict)):
        try:
            draws = draws()
        except Exception:  # noqa: BLE001
            pass

    # File path.
    if isinstance(draws, (str, os.PathLike)):
        path = os.fspath(draws)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference draws file not found: {path}")
        if path.endswith(".npz"):
            data = np.load(path)
            # Use first array in the archive.
            arr = next(v for _, v in data.items())
            return np.asarray(arr)
        if path.endswith(".csv"):
            arr = np.genfromtxt(path, delimiter=",")
        else:
            arr = np.genfromtxt(path)
        return np.asarray(arr, dtype=float)

    # Dictionary of named draws (posteriordb sometimes returns these).
    if isinstance(draws, dict):
        cols = [np.asarray(v, dtype=float) for v in draws.values()]
        if not cols:
            return None
        n = min(len(c) for c in cols)
        return np.stack([c[:n] for c in cols], axis=1)

    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _bridgestan_model_from_posterior(
    posterior: Any, cache_dir: str
) -> Tuple[Any, str, Optional[str]]:
    """Compile/load a BridgeStan model from a posteriordb posterior.

    Returns ``(model, stan_path, data_json_path_or_none)``.
    """
    _require_bridgestan()
    code = _model_code(posterior)
    stan_path = _write_stan_model(code, cache_dir, posterior.name or "model")

    data = _model_data(posterior)
    data_path: Optional[str] = None
    if isinstance(data, dict):
        import json

        data_path = os.path.join(cache_dir, f"{posterior.name}.json")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    elif isinstance(data, str) and os.path.exists(data):
        data_path = data

    model = bridgestan.StanModel(
        stan_path, data_path if data_path is not None else "", seed=1234
    )
    return model, stan_path, data_path


def load_posterior_target(
    name: str,
    *,
    pdb_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    load_reference_draws: bool = True,
) -> PosteriorTarget:
    """Load a Section 5.2 posterior target from posteriordb + BridgeStan.

    Parameters
    ----------
    name:
        Model alias, e.g. ``"ark"``, ``"gp-pois-regr"`` or
        ``"eight-schools-centered"``.
    pdb_path:
        Optional path to a local posteriordb checkout/database. When ``None``,
        the package's default database location is used.
    cache_dir:
        Directory used to cache extracted Stan files and compiled models.
        Defaults to ``~/.cache/bam/posteriordb``.
    load_reference_draws:
        If ``True`` (default), attempt to load the posterior's HMC reference
        draws for posterior mean/SD diagnostics.
    """
    _require_posteriordb()
    _require_bridgestan()

    canonical = canonical_model_name(name)
    spec = POSTERIOR_SPECS[canonical]

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "bam", "posteriordb"
        )
    os.makedirs(cache_dir, exist_ok=True)

    # Construct posteriordb database.
    if pdb_path is None:
        try:
            pdb = posteriordb.PosteriorDatabase()
        except Exception:  # noqa: BLE001 - retry without arguments
            pdb = posteriordb.PosteriorDatabase()
    else:
        pdb = posteriordb.PosteriorDatabase(pdb_path)

    posterior = _find_posterior_by_aliases(pdb, name)

    # BridgeStan model.
    model, _stan_path, _data_path = _bridgestan_model_from_posterior(
        posterior, cache_dir
    )

    def _log_density(theta: jnp.ndarray) -> float:
        theta_np = _as_numpy(theta)
        return float(model.log_density(theta_np))

    def _log_density_grad(theta: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        theta_np = _as_numpy(theta)
        lp, grad = model.log_density_gradient(theta_np)
        return float(lp), jnp.asarray(grad, dtype=jnp.result_type(float))

    # Reference draws.
    ref: Optional[np.ndarray] = None
    if load_reference_draws:
        try:
            if hasattr(posterior, "reference_draws"):
                raw = posterior.reference_draws()
                ref = _load_reference_draws(raw)
            elif hasattr(posterior, "reference_draw"):
                raw = posterior.reference_draw()
                ref = _load_reference_draws(raw)
        except Exception as exc:  # noqa: BLE001 - reference draws optional
            print(f"[bam.posterior_targets] Warning: could not load reference "
                  f"draws for {name!r}: {exc}")

    # Sanity-check dimension against BridgeStan parameter count if possible.
    if hasattr(model, "param_unc_num"):
        d = int(model.param_unc_num())
        if d != spec.dim:
            print(
                f"[bam.posterior_targets] Warning: BridgeStan reports D={d} "
                f"for {name!r}, registry says D={spec.dim}. Using {d}."
            )
            spec_dim = d
        else:
            spec_dim = spec.dim
    else:
        spec_dim = spec.dim

    return PosteriorTarget(
        name=canonical,
        dim=spec_dim,
        log_density_fn=_log_density,
        log_density_grad_fn=_log_density_grad,
        reference_draws=None if ref is None else jnp.asarray(ref),
    )


def make_posterior_target(
    name: str,
    *,
    log_density_fn: Optional[Callable[[jnp.ndarray], float]] = None,
    log_density_grad_fn: Optional[
        Callable[[jnp.ndarray], Tuple[float, jnp.ndarray]]
    ] = None,
    reference_draws: Optional[jnp.ndarray] = None,
    pdb_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    load_reference_draws: bool = True,
) -> PosteriorTarget:
    """Flexible posterior-target factory.

    If explicit ``log_density_fn`` and ``log_density_grad_fn`` are provided,
    they are wrapped directly (useful for tests and custom models). Otherwise
    this delegates to :func:`load_posterior_target` which uses
    posteriordb + BridgeStan.
    """
    if log_density_fn is not None:
        canonical = canonical_model_name(name)
        spec = POSTERIOR_SPECS[canonical]
        ref = (
            jnp.asarray(reference_draws)
            if reference_draws is not None
            else None
        )
        return PosteriorTarget(
            name=canonical,
            dim=spec.dim,
            log_density_fn=log_density_fn,
            log_density_grad_fn=log_density_grad_fn,
            reference_draws=ref,
        )
    return load_posterior_target(
        name,
        pdb_path=pdb_path,
        cache_dir=cache_dir,
        load_reference_draws=load_reference_draws,
    )


def posterior_relative_errors(
    mu: jnp.ndarray,
    sd: jnp.ndarray,
    reference_draws: jnp.ndarray,
) -> Tuple[float, float]:
    """Compute posterior mean and SD relative errors against HMC draws.

    The paper reports ``||mu - mu_hat|| / ||mu_hat||`` and
    ``||sd - sd_hat|| / ||sd_hat||`` (Euclidean norms), where ``mu_hat`` and
    ``sd_hat`` are the reference-draw moment estimates.
    """
    mu = jnp.asarray(mu)
    sd = jnp.asarray(sd)
    ref = jnp.asarray(reference_draws)
    if ref.ndim != 2:
        raise ValueError("reference_draws must be a 2-D array of shape (N, D)")

    mu_hat = jnp.mean(ref, axis=0)
    sd_hat = jnp.std(ref, axis=0, ddof=0)

    mu_err = jnp.linalg.norm(mu - mu_hat) / (jnp.linalg.norm(mu_hat) + 1e-12)
    sd_err = jnp.linalg.norm(sd - sd_hat) / (jnp.linalg.norm(sd_hat) + 1e-12)
    return float(mu_err), float(sd_err)


def available_posteriors() -> list[str]:
    """Return the canonical posterior names supported by this module."""
    return sorted(POSTERIOR_SPECS)


__all__ = [
    "PosteriorSpec",
    "POSTERIOR_SPECS",
    "PosteriorTarget",
    "canonical_model_name",
    "load_posterior_target",
    "make_posterior_target",
    "posterior_relative_errors",
    "available_posteriors",
]
