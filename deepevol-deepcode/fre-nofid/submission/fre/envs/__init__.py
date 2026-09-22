"""Environment wrappers for the FRE reproduction.

This package aggregates the three domain wrappers used for zero-shot
evaluation of the FRE latent policy:

* :mod:`fre.envs.antmaze_wrapper` -- AntMaze (``antmaze-large-diverse-v2``)
  with 32-bin XY discretisation, centre start and the paper's goal /
  directional / opensimplex / corridor task families.
* :mod:`fre.envs.exorl_wrapper` -- ExORL RND walker & cheetah, with physics
  features appended to the *encoder* observation only, per-dimension
  std-normalisation and the goal (dist < 0.1) / velocity task families.
* :mod:`fre.envs.kitchen_wrapper` -- D4RL Kitchen with the seven standard
  sparse subtasks read from the last seven observation dims.

All wrappers share a small gym-like interface
(``reset`` / ``step`` / ``set_task`` / ``eval_tasks`` / ``sample_context``)
so that :mod:`fre.evaluate` can treat them uniformly.

Imports are performed defensively (see :func:`_try_import`) so that a partial
install -- for example a machine without ``d4rl``/``mujoco`` -- can still
import the package and use the offline dataset-replay fallbacks.
"""

from __future__ import annotations

# Re-exported symbols are populated dynamically below; keep the list explicit
# so that tooling and downstream code can introspect the public surface.
from typing import Any, Dict, List, Optional  # noqa: F401

__all__: List[str] = []


def _try_import(module: str, names: List[str]) -> None:
    """Best-effort re-export of ``names`` from ``module``.

    Any failure (missing optional dependency, absent symbol, syntax error in a
    partially-written module) is swallowed so that the remaining wrappers stay
    importable.
    """

    try:
        mod = __import__(module, fromlist=list(names))
    except Exception:  # pragma: no cover - defensive
        return
    for name in names:
        try:
            value = getattr(mod, name)
        except AttributeError:
            continue
        globals()[name] = value
        if name not in __all__:
            __all__.append(name)


# ---------------------------------------------------------------------------
# AntMaze
# ---------------------------------------------------------------------------
_try_import(
    "fre.envs.antmaze_wrapper",
    [
        # constants
        "ANTMAZE_DATASET",
        "RAW_OBS_DIM",
        "ACTION_DIM",
        "XY_DIM",
        "MAX_EPISODE_STEPS",
        "NUM_XY_BINS",
        "DEFAULT_XY_EXTENT",
        "DEFAULT_GOAL_DISTANCE",
        "GOAL_TASK_GOALS",
        "DIRECTIONAL_GOALS",
        "SIMPLEX_SEEDS",
        "PATH_TASKS",
        "TASK_FAMILIES",
        # classes
        "AntMazeWrapper",
        "AntMazeConfig",
        "AntMazeTaskSpec",
        "AntMazeGoalTask",
        "AntMazeDirectionalTask",
        "AntMazeSimplexTask",
        "AntMazePathTask",
        # factories / helpers
        "make_antmaze_env",
        "antmaze_eval_tasks",
        "goal_task",
        "directional_task",
        "simplex_task",
        "path_task",
        "goal_reward",
        "goal_done",
        "directional_reward",
        "directional_done",
        "simplex_reward",
        "path_reward",
        "path_done",
        "path_route",
        "discretize_xy",
        "bin_to_xy",
        "xy_distance",
        "bin_distance",
    ],
)

# ---------------------------------------------------------------------------
# ExORL (walker / cheetah)
# ---------------------------------------------------------------------------
_try_import(
    "fre.envs.exorl_wrapper",
    [
        "ExORLWrapper",
        "ExORLConfig",
        "ExORLTaskSpec",
        "ExORLGoalTask",
        "ExORLVelocityTask",
        "make_exorl_env",
        "make_exorl_envs",
        "exorl_eval_tasks",
        "LIVE_ENV_CANDIDATES",
    ],
)

# ---------------------------------------------------------------------------
# Kitchen
# ---------------------------------------------------------------------------
_try_import(
    "fre.envs.kitchen_wrapper",
    [
        # constants
        "RAW_OBS_DIM",
        "ACTION_DIM",
        "NUM_SUBTASKS",
        "MAX_EPISODE_STEPS",
        "FLAG_THRESHOLD",
        "SUBTASK_ORDER",
        "SUBTASK_FLAG_INDEX",
        "SUBTASK_ALIASES",
        "SUBTASK_DISPLAY",
        "KITCHEN_EVAL_TASK_NAMES",
        "DEFAULT_DATASET",
        # classes
        "KitchenWrapper",
        "KitchenTaskSpec",
        "KitchenConfig",
        "KitchenSubtaskEnv",
        # factories / helpers
        "make_kitchen_env",
        "kitchen_eval_tasks",
        "resolve_subtask",
        "subtask_flags",
        "subtask_complete",
        "all_subtask_flags",
        "sparse_subtask_reward",
        "goal_style_subtask_reward",
        "episode_ends_from_dataset",
        "episode_slices",
    ],
)

# ---------------------------------------------------------------------------
# Domain <-> wrapper plumbing
# ---------------------------------------------------------------------------

DOMAINS = ("antmaze", "exorl", "kitchen")

#: ExORL is a multi-domain dataset; each sub-domain is an independent env.
DOMAIN_SUBDOMAINS: Dict[str, tuple] = {"exorl": ("walker", "cheetah")}

#: Canonical dataset identifiers used by the loaders / wrappers.
DEFAULT_DATASETS: Dict[str, str] = {
    "antmaze": "antmaze-large-diverse-v2",
    "kitchen": "kitchen-complete-v0",
    "exorl": "rnd",
}

#: Max episode length per domain (paper / D4RL conventions).
MAX_EPISODE_STEPS_BY_DOMAIN: Dict[str, int] = {
    "antmaze": 2000,
    "exorl": 1000,
    "kitchen": 1000,
}

for _name in (
    "DOMAINS",
    "DOMAIN_SUBDOMAINS",
    "DEFAULT_DATASETS",
    "MAX_EPISODE_STEPS_BY_DOMAIN",
):
    if _name not in __all__:
        __all__.append(_name)


def _normalise_domain(domain: str) -> str:
    """Map a free-form domain / sub-domain string onto a canonical domain."""

    if domain is None:
        raise ValueError("domain must be provided (one of %s)" % (DOMAINS,))
    key = str(domain).strip().lower().replace("_", "-")
    if key.startswith("ant"):
        return "antmaze"
    if key.startswith("kitchen"):
        return "kitchen"
    if key.startswith("exorl") or key in ("walker", "cheetah"):
        return "exorl"
    raise ValueError(
        "Unknown domain %r; expected one of %s" % (domain, DOMAINS)
    )


def sub_domain_of(domain: str, dataset: Any = None) -> Optional[str]:
    """Return the ExORL sub-domain (``walker`` / ``cheetah``) if relevant.

    ``None`` is returned for single-domain datasets such as AntMaze/Kitchen.
    """

    canon = _normalise_domain(domain)
    if canon != "exorl":
        return None
    for candidate in ("walker", "cheetah"):
        if isinstance(domain, str) and candidate in domain.lower():
            return candidate
    if isinstance(dataset, str):
        for candidate in ("walker", "cheetah"):
            if candidate in dataset.lower():
                return candidate
    return None


def make_env(
    domain: str = "antmaze",
    task: Any = None,
    *,
    dataset: Any = None,
    sub_domain: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create an FRE environment wrapper for ``domain``.

    Parameters
    ----------
    domain:
        ``"antmaze"``, ``"exorl"``, ``"kitchen"`` (prefix matching is allowed,
        e.g. ``"exorl-walker"``).
    task:
        Optional task descriptor / name to pin the wrapper to a single task.
    dataset:
        Optional dataset name or a pre-loaded dataset payload.  For ExORL this
        may also be the sub-domain name.
    sub_domain:
        Explicit ExORL sub-domain (``walker`` / ``cheetah``) when ``domain``
        does not encode it.
    **kwargs:
        Forwarded to the wrapper factory.

    Returns
    -------
    One of :class:`AntMazeWrapper`, :class:`ExORLWrapper`,
    :class:`KitchenWrapper` (or a task-pinned subclass).
    """

    canon = _normalise_domain(sub_domain or domain)

    if canon == "antmaze":
        if "make_antmaze_env" not in globals():  # pragma: no cover - defensive
            raise ImportError(
                "fre.envs.antmaze_wrapper is unavailable; cannot build an "
                "AntMaze environment."
            )
        kwargs.setdefault("dataset", dataset)
        return make_antmaze_env(task=task, **kwargs)

    if canon == "exorl":
        if "make_exorl_env" not in globals():  # pragma: no cover - defensive
            raise ImportError(
                "fre.envs.exorl_wrapper is unavailable; cannot build an "
                "ExORL environment."
            )
        resolved = sub_domain or sub_domain_of(domain, dataset) or "walker"
        kwargs.setdefault("root", kwargs.pop("root", None))
        return make_exorl_env(domain=resolved, task=task, **kwargs)

    if canon == "kitchen":
        if "make_kitchen_env" not in globals():  # pragma: no cover - defensive
            raise ImportError(
                "fre.envs.kitchen_wrapper is unavailable; cannot build a "
                "Kitchen environment."
            )
        kwargs.setdefault("dataset", dataset)
        return make_kitchen_env(task=task, **kwargs)

    raise ValueError("Unknown domain %r" % (domain,))  # pragma: no cover


def make_envs(domains: Any = DOMAINS, **kwargs: Any) -> Dict[str, Any]:
    """Create a mapping ``{env_key: wrapper}`` for several domains.

    ExORL is expanded into its ``walker`` / ``cheetah`` sub-domains, yielding
    keys such as ``"exorl-walker"``.
    """

    if isinstance(domains, str):
        domains = (domains,)
    envs: Dict[str, Any] = {}
    for domain in domains:
        canon = _normalise_domain(domain)
        if canon == "exorl":
            subs = DOMAIN_SUBDOMAINS["exorl"]
            if sub_domain_of(domain) is not None:
                subs = (sub_domain_of(domain),)
            for sub in subs:
                envs["exorl-%s" % sub] = make_env(
                    "exorl", sub_domain=sub, **kwargs
                )
        else:
            envs[canon] = make_env(canon, **kwargs)
    return envs


def eval_tasks_for(
    domain: str,
    env: Any = None,
    *,
    dataset: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return the zero-shot evaluation task suite for ``domain``.

    Uses the wrapper-specific ``*_eval_tasks`` helper, mirroring
    :func:`fre.evaluate.suite_for_domain`.
    """

    canon = _normalise_domain(domain)
    if canon == "antmaze" and "antmaze_eval_tasks" in globals():
        return antmaze_eval_tasks(env, **kwargs) if env is not None else antmaze_eval_tasks(**kwargs)
    if canon == "exorl" and "exorl_eval_tasks" in globals():
        resolved = sub_domain_of(domain, dataset) or "walker"
        return exorl_eval_tasks(resolved, dataset=dataset, **kwargs)
    if canon == "kitchen" and "kitchen_eval_tasks" in globals():
        return kitchen_eval_tasks(env, **kwargs) if env is not None else kitchen_eval_tasks(**kwargs)
    return {}


for _fn in ("make_env", "make_envs", "eval_tasks_for", "sub_domain_of"):
    if _fn not in __all__:
        __all__.append(_fn)


def __getattr__(name: str) -> Any:
    """Provide an informative error for unknown attributes."""

    raise AttributeError(
        "module %r has no attribute %r. Available names: %s"
        % (__name__, name, ", ".join(sorted(__all__)))
    )


def __dir__() -> List[str]:  # pragma: no cover - introspection helper
    return sorted(set(globals()) | set(__all__))
