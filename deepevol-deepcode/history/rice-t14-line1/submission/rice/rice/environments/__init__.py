"""Environment suite used by RICE.

This sub-package provides thin, dependency-tolerant wrappers around every
application used in the paper (Appendix C.2):

======================  ====================================================
File                    Environments
======================  ====================================================
``mujoco_dense.py``     ``Hopper-v3``, ``Walker2d-v3``, ``Reacher-v2``,
                        ``HalfCheetah-v3`` (dense reward)
``mujoco_sparse.py``    ``SparseHopper`` (x > 0.6) and
                        ``SparseHalfCheetah`` (x > 5), Mazoure et al. (2019)
``selfish_mining.py``   Bar-Zur et al. (2023) blockchain model with actions
                        Adopt l / Reveal l / Mine, 4x128 MLP policy network
``cage_challenge2.py``  TTCP CAGE Challenge 2 blue agent (Cardiff champion)
                        against the red agent "B-line", trials 30/50/100
``autodriving.py``      MetaDrive "Macro-v1" (DI-drive), action a in [-1, 1]^2
======================  ====================================================

Design notes
------------
* Nothing here imports MuJoCo / CAGE / MetaDrive at *package import* time.  Every
  environment module is imported lazily inside :func:`make_env` / the registry
  helpers, so ``import rice`` keeps working on a machine that only has the
  algorithm dependencies installed (see ``_common.py``).
* :data:`ENV_REGISTRY` maps the short names used throughout the paper (and in
  ``configs/*.yaml``) onto ``(module, factory attribute)`` pairs.
* :func:`make_env` returns a *single* environment instance; it never wraps it in
  a vector env, because RICE's Algorithm 1/2 loops are inherently sequential
  (one ``s_0`` per outer iteration, Go-Explore style state restore).

Only the in-scope environments listed in the reproduction plan are exposed here;
the malware-mutation module is intentionally absent (out of scope).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._common import (  # noqa: F401  (re-exported for convenience)
    EnvBase,
    GYM_AVAILABLE,
    IS_GYMNASIUM,
    WrapperBase,
    env_max_episode_steps,
    make_box,
    make_discrete,
    normalize_reset,
    normalize_step,
)

__all__ = [
    # shared plumbing
    "EnvBase",
    "WrapperBase",
    "make_box",
    "make_discrete",
    "normalize_reset",
    "normalize_step",
    "env_max_episode_steps",
    "GYM_AVAILABLE",
    "IS_GYMNASIUM",
    # registry / factories
    "ENV_REGISTRY",
    "ENV_ALIASES",
    "DENSE_ENVS",
    "SPARSE_ENVS",
    "ALL_ENVS",
    "register_env",
    "available_envs",
    "make_env",
    "get_env_spec",
    "default_net_arch",
    "make_dense_env",
    "make_sparse_env",
    "make_selfish_mining_env",
    "make_cage_env",
    "make_autodriving_env",
    "load_env_module",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# canonical name -> (module name, factory attribute, human readable description)
ENV_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    # ---- dense MuJoCo (Appendix C.2) ------------------------------------
    "Hopper-v3": (
        "mujoco_dense",
        "make_hopper",
        "Hopper 3-DoF hopper, dense reward = healthy + forward + control cost",
    ),
    "Walker2d-v3": (
        "mujoco_dense",
        "make_walker2d",
        "Walker2d six-hinge walker, observations normalized, dense reward",
    ),
    "Reacher-v2": (
        "mujoco_dense",
        "make_reacher",
        "Reacher two-jointed arm, reward distance - reward control, 50-step episodes",
    ),
    "HalfCheetah-v3": (
        "mujoco_dense",
        "make_halfcheetah",
        "HalfCheetah 9-link cheetah, observations normalized, dense reward",
    ),
    # ---- sparse MuJoCo (Mazoure et al. 2019) ----------------------------
    "SparseHopper": (
        "mujoco_sparse",
        "make_sparse_hopper",
        "Hopper with sparse reward informing x only if x > 0.6",
    ),
    "SparseHalfCheetah": (
        "mujoco_sparse",
        "make_sparse_halfcheetah",
        "HalfCheetah with sparse reward informing x only if x > 5",
    ),
    "SparseWalker2d": (
        "mujoco_sparse",
        "make_sparse_walker2d",
        "Walker2d with sparse reward (OUT OF SCOPE for refining, kept for completeness)",
    ),
    # ---- non-MuJoCo applications ----------------------------------------
    "SelfishMining": (
        "selfish_mining",
        "make_selfish_mining",
        "Bar-Zur et al. blockchain selfish-mining env (Adopt l / Reveal l / Mine)",
    ),
    "CageChallenge2": (
        "cage_challenge2",
        "make_cage_challenge2",
        "TTCP CAGE-2 blue agent vs red 'B-line', trials 30/50/100",
    ),
    "Macro-v1": (
        "autodriving",
        "make_autodriving",
        "MetaDrive 'Macro-v1' via DI-drive, action a in [-1, 1]^2",
    ),
}

# Friendly aliases (case/format insensitive).  ``ENV_ALIASES`` maps a normalized
# key onto a canonical registry key.
ENV_ALIASES: Dict[str, str] = {
    "hopper": "Hopper-v3",
    "hopper-v3": "Hopper-v3",
    "walker2d": "Walker2d-v3",
    "walker2d-v3": "Walker2d-v3",
    "reacher": "Reacher-v2",
    "reacher-v2": "Reacher-v2",
    "halfcheetah": "HalfCheetah-v3",
    "halfcheetah-v3": "HalfCheetah-v3",
    "sparsehopper": "SparseHopper",
    "sparse-hopper": "SparseHopper",
    "sparsehalfcheetah": "SparseHalfCheetah",
    "sparse-halfcheetah": "SparseHalfCheetah",
    "sparsewalker2d": "SparseWalker2d",
    "sparse-walker2d": "SparseWalker2d",
    "selfish": "SelfishMining",
    "selfishmining": "SelfishMining",
    "selfish_mining": "SelfishMining",
    "cage": "CageChallenge2",
    "cage2": "CageChallenge2",
    "cagechallenge2": "CageChallenge2",
    "cage_challenge2": "CageChallenge2",
    "auto": "Macro-v1",
    "autodriving": "Macro-v1",
    "autonomousdriving": "Macro-v1",
    "macro-v1": "Macro-v1",
    "macrov1": "Macro-v1",
}

#: Environments trained with a dense reward in Experiment II/III.
DENSE_ENVS: Tuple[str, ...] = (
    "Hopper-v3",
    "Walker2d-v3",
    "Reacher-v2",
    "HalfCheetah-v3",
    "SelfishMining",
    "CageChallenge2",
    "Macro-v1",
)

#: Environments with a sparse reward (refining curves, Figure 2).  Only
#: SparseHopper and SparseHalfCheetah are in scope per the addendum.
SPARSE_ENVS: Tuple[str, ...] = ("SparseHopper", "SparseHalfCheetah")

#: Everything exposed by this sub-package.
ALL_ENVS: Tuple[str, ...] = tuple(ENV_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Network architectures (Appendix C.2, addendum "Architectures")
# ---------------------------------------------------------------------------
# The mask network must mirror the target agent's architecture.  MuJoCo agents
# are trained with the Stable-Baselines3 default ``MlpPolicy`` (two hidden
# layers of 64 units, tanh).
_NET_ARCH: Dict[str, Tuple[int, ...]] = {
    "Hopper-v3": (64, 64),
    "Walker2d-v3": (64, 64),
    "Reacher-v2": (64, 64),
    "HalfCheetah-v3": (64, 64),
    "SparseHopper": (64, 64),
    "SparseHalfCheetah": (64, 64),
    "SparseWalker2d": (64, 64),
    # Selfish Mining: 4-layer MLP with hidden size 128, 128, 128, 128 (§C.2).
    "SelfishMining": (128, 128, 128, 128),
    # CAGE Challenge 2: MLP with hidden size 64, 64, 64 (§C.2).
    "CageChallenge2": (64, 64, 64),
    # Autonomous driving: DI-engine VAC default template (§C.2 / addendum).
    "Macro-v1": (256, 256),
}

#: Observation normalization required when training the DRL agent (§C.2).
_NORMALIZE_OBS = {
    "Walker2d-v3": True,
    "HalfCheetah-v3": True,
    "Macro-v1": True,
}


def _normalize_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def resolve_env_name(name: str) -> str:
    """Resolve a canonical name or friendly alias to a registry key.

    Raises ``KeyError`` with a helpful message listing the known names.
    """
    if name in ENV_REGISTRY:
        return name
    key = _normalize_key(name)
    if key in ENV_ALIASES:
        return ENV_ALIASES[key]
    # last resort: case-insensitive match on the canonical names
    for canonical in ENV_REGISTRY:
        if _normalize_key(canonical) == key:
            return canonical
    raise KeyError(
        "Unknown environment {!r}. Known environments: {}. Known aliases: {}".format(
            name, sorted(ENV_REGISTRY), sorted(ENV_ALIASES)
        )
    )


def available_envs() -> List[str]:
    """Return the canonical names of every registered environment."""
    return sorted(ENV_REGISTRY)


def register_env(
    name: str,
    module: str,
    factory: str,
    description: str = "",
    aliases: Optional[Tuple[str, ...]] = None,
) -> None:
    """Register an extra environment (used by tests / user extensions)."""
    ENV_REGISTRY[name] = (module, factory, description)
    for alias in aliases or ():
        ENV_ALIASES[_normalize_key(alias)] = name


def get_env_spec(name: str) -> Dict[str, Any]:
    """Return metadata (name, module, factory, net_arch, ...) for ``name``."""
    canonical = resolve_env_name(name)
    module, factory, description = ENV_REGISTRY[canonical]
    return {
        "name": canonical,
        "module": module,
        "factory": factory,
        "description": description,
        "net_arch": _NET_ARCH.get(canonical, (64, 64)),
        "normalize_obs": _NORMALIZE_OBS.get(canonical, False),
        "sparse": canonical in SPARSE_ENVS,
    }


def default_net_arch(name: str) -> Tuple[int, ...]:
    """Target-agent / mask-network architecture for ``name`` (§C.2)."""
    return tuple(_NET_ARCH.get(resolve_env_name(name), (64, 64)))


def load_env_module(module: str):
    """Import an environment module by its short name.

    Tries ``rice.environments.<module>`` first and falls back to a relative
    import so that the package works whether or not the top-level ``rice``
    package is installed.
    """
    try:
        return importlib.import_module("{}.{}".format(__name__, module))
    except Exception:
        try:
            return importlib.import_module("." + module, __package__)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Could not import environment module {!r} ({})".format(module, exc)
            ) from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_env(name: str, **kwargs: Any):
    """Create a single RICE environment.

    Parameters
    ----------
    name:
        Canonical environment name or friendly alias (see :data:`ENV_ALIASES`).
    **kwargs:
        Forwarded verbatim to the environment factory.  Common options:

        ``seed``            seed the environment RNG,
        ``normalize``       wrap observations in a running-mean/std normalizer,
        ``sparse``          force the sparse-reward variant (MuJoCo only),
        ``max_episode_steps`` episode length override,
        ``render_mode``     forwarded to ``gym.make``,
        ``reset_to``        Go-Explore style initial state (see ``env_reset``).

    Returns
    -------
    gym.Env (or the wrapper stack) ready for ``reset`` / ``step``.
    """
    spec = get_env_spec(name)
    module = load_env_module(spec["module"])
    factory: Callable[..., Any] = getattr(module, spec["factory"])
    return factory(**kwargs)


def _call_first_available(module: Any, candidates: Tuple[str, ...], **kwargs: Any):
    """Call the first existing factory in ``candidates`` (naming tolerance)."""
    for attr in candidates:
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn(**kwargs)
    raise AttributeError(
        "Module {!r} exposes none of {}; it probably could not be imported "
        "because an optional simulator dependency is missing.".format(
            getattr(module, "__name__", module), candidates
        )
    )


def make_dense_env(name: str = "Hopper-v3", **kwargs: Any):
    """Build one of the dense MuJoCo tasks (§C.2)."""
    module = load_env_module("mujoco_dense")
    return _call_first_available(
        module,
        ("make_mujoco_dense", "make_env", "make_dense_env"),
        name=name,
        **kwargs,
    )


def make_sparse_env(name: str = "SparseHopper", **kwargs: Any):
    """Build one of the sparse-reward MuJoCo tasks (Mazoure et al. 2019)."""
    module = load_env_module("mujoco_sparse")
    return _call_first_available(
        module,
        ("make_mujoco_sparse", "make_env", "make_sparse_env"),
        name=name,
        **kwargs,
    )


def make_selfish_mining_env(**kwargs: Any):
    """Build the Bar-Zur et al. (2023) selfish-mining environment (§C.2)."""
    module = load_env_module("selfish_mining")
    return _call_first_available(
        module,
        ("make_selfish_mining", "make_env", "SelfishMiningEnv"),
        **kwargs,
    )


def make_cage_env(**kwargs: Any):
    """Build the CAGE Challenge 2 environment (Cardiff champion, §C.2)."""
    module = load_env_module("cage_challenge2")
    return _call_first_available(
        module,
        ("make_cage_challenge2", "make_env", "CageChallenge2Env"),
        **kwargs,
    )


def make_autodriving_env(**kwargs: Any):
    """Build MetaDrive "Macro-v1" through DI-drive (§C.2)."""
    module = load_env_module("autodriving")
    return _call_first_available(
        module,
        ("make_autodriving", "make_env", "AutoDrivingEnv"),
        **kwargs,
    )
