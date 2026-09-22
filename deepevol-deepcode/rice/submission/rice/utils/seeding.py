"""Reproducibility helpers for the RICE codebase.

The paper trains the mask network (Stage 1) and the refining policy
(Stage 2) over several seeds and reports ``mean +- std`` over 3 seeds
(Experiment I runs ``500 trajectories x 3 seeds``).  To make those
aggregations meaningful every component of the pipeline seeds the RNGs
through the helpers in this module.

The functions are intentionally dependency-light (``numpy`` + the
standard library + an *optional* torch import) so that they can be used
from every sub-package of ``rice`` (envs, explanation, refining,
baselines, scripts, experiments).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["set_seed", "get_rng", "seed_env", "seed_action_space", "seed_from"]

# ---------------------------------------------------------------------------
# Optional torch support (torch is a hard dependency of the project, but the
# utils module must stay importable in minimal environments, e.g. when only
# the environment factory is being smoke-tested).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly when torch is installed
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and (if available) torch.

    Parameters
    ----------
    seed:
        The seed value.  ``None`` is not accepted here on purpose: the
        plan calls for *fixed* seeds per run so that the ``mean +- std``
        over the 3 reported seeds is reproducible.
    deterministic:
        Forwarded to :func:`torch.use_deterministic_algorithms` when
        torch is available.  Defaults to ``False`` because MuJoCo
        environments are not fully deterministic and Sliding-window
        fidelity evaluation (Experiment I) does not require bit-exact
        reproducibility.

    Returns
    -------
    int
        The seed that was applied (convenient for logging).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _HAS_TORCH:  # pragma: no cover
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    return seed


def get_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Return a dedicated :class:`numpy.random.RandomState`.

    Using an explicit RNG object (instead of the global numpy state) keeps
    e.g. the mixed-initial-state sampler in Algorithm 2 independent from the
    action noise of the environments.
    """
    return np.random.RandomState(seed)


def seed_env(env, seed: Optional[int] = None, rank: int = 0) -> Optional[int]:
    """Best-effort seeding of a (possibly vectorised) gym environment.

    Works with single gym envs, SB3 ``VecEnv``s and the wrapper stack used
    by ``rice.envs`` (sparse reward / reset wrapper / normalizer).  The
    function walks ``env.env`` chains until it finds a ``seed`` attribute or
    gives up, and never raises.
    """
    if env is None:
        return None
    seed = None if seed is None else int(seed) + int(rank)

    for attr in ("seed", "seed_"):
        fn = getattr(env, attr, None)
        if callable(fn):
            try:
                fn(seed)
                return seed
            except Exception:
                pass

    inner = getattr(env, "env", None)
    if inner is not None and inner is not env:
        return seed_env(inner, seed, rank)
    return None


def seed_action_space(env, seed: Optional[int] = None) -> Optional[int]:
    """Seed the action space of ``env`` when it supports it.

    Used by the fidelity evaluator: after fast-forwarding to a critical state
    we take *random* actions for ``l`` steps, so the action-space RNG has to
    be controllable (``K = 10/20/30/40%`` windows are compared under the same
    seeds).
    """
    if env is None:
        return None
    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "seed"):
        try:
            space.seed(seed)
            return seed
        except Exception:
            pass
    return seed_env(env, seed)


def seed_from(base_seed: int, *offsets: int) -> int:
    """Deterministically derive a child seed from ``base_seed``.

    Example: ``seed_from(0, 1, 7)`` gives the seed of the 7-th trajectory of
    the 1-st of the 3 Experiment-I seeds.
    """
    value = int(base_seed) & 0xFFFFFFFF
    for off in offsets:
        value = (value * 1000003 + int(off) + 0x9E3779B9) & 0xFFFFFFFF
    return int(value)
