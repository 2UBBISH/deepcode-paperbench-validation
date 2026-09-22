"""Benchmark and scientific simulator tasks for Simformer.

This package collects every simulator used in the paper (Appendix A2.2 and
Sections 4.2-4.4).  All tasks share a small common interface so that the
training / sampling / evaluation scripts can treat them uniformly::

    task = get_task("two_moons")            # or build_task("slcp", seed=1)
    theta, x = task(n_samples, rng)         # simulate a joint sample
    joint = task.to_joint(theta, x)
    mask = task.posterior_condition_mask()  # M_C for p(theta|x)

The shared interface is implemented by :class:`TaskBase`.  Sub-modules are
imported lazily (see :func:`load_task_module`) so that this package can be
imported even when optional third-party dependencies of a single task (for
example torch for the embedding network of the gravitational-wave task) are
missing.

Task name mapping (plan "Task simulators", Sources: Sec. A2.2, 4.2-4.4,
Sec. A3.2 and Addendum "Tasks"):

=============================  =========================  =================
name                           module                     params / data
=============================  =========================  =================
``gaussian_linear``            gaussian_linear.py         10 / 10
``gaussian_mixture``           gaussian_mixture.py         2 / 2
``two_moons``                  two_moons.py                2 / 2
``slcp``                       slcp.py                     5 / 8
``tree``                       tree.py                     3 / 4
``hmm``                        hmm.py                     10 / 10
``lotka_volterra``             lotka_volterra.py           4 / 2*T
``sird``                       sird.py                     3+ / 4*T
``hodgkin_huxley``             hodgkin_huxley.py           4 / 1*T (+7 stats)
``gravitational_waves``        gravitational_waves.py      2 / 2 * 8192
=============================  =========================  =================
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np

__all__ = [
    "TaskBase",
    "TaskSpec",
    "TASK_MODULES",
    "TASK_ALIASES",
    "register_task",
    "available_tasks",
    "load_task_module",
    "task_class",
    "get_task",
    "build_task",
    "task_attention_mask",
    "task_dimensions",
    "attach_attention_mask",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Canonical task name -> fully qualified module path.
TASK_MODULES: Dict[str, str] = {
    "gaussian_linear": "simformer.tasks.gaussian_linear",
    "gaussian_mixture": "simformer.tasks.gaussian_mixture",
    "two_moons": "simformer.tasks.two_moons",
    "slcp": "simformer.tasks.slcp",
    "tree": "simformer.tasks.tree",
    "hmm": "simformer.tasks.hmm",
    "lotka_volterra": "simformer.tasks.lotka_volterra",
    "sird": "simformer.tasks.sird",
    "hodgkin_huxley": "simformer.tasks.hodgkin_huxley",
    "gravitational_waves": "simformer.tasks.gravitational_waves",
}

#: Alternative spellings used in the paper / scripts.
TASK_ALIASES: Dict[str, str] = {
    "linear_gaussian": "gaussian_linear",
    "gaussian-linear": "gaussian_linear",
    "linear-gaussian": "gaussian_linear",
    "gaussian_mixtures": "gaussian_mixture",
    "gaussian-mixture": "gaussian_mixture",
    "mixture": "gaussian_mixture",
    "twomoons": "two_moons",
    "two-moons": "two_moons",
    "two_moon": "two_moons",
    "slcp_task": "slcp",
    "lotka-volterra": "lotka_volterra",
    "lv": "lotka_volterra",
    "lotkavolterra": "lotka_volterra",
    "sir": "sird",
    "sird_task": "sird",
    "hh": "hodgkin_huxley",
    "hodgkin-huxley": "hodgkin_huxley",
    "hodgkinhuxley": "hodgkin_huxley",
    "gw": "gravitational_waves",
    "gravitational-waves": "gravitational_waves",
    "gravitationalwaves": "gravitational_waves",
}

#: Lazily populated cache of loaded task modules.
_MODULE_CACHE: Dict[str, Any] = {}

#: Extra user registered tasks: name -> callable returning the task object.
_USER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_task(name: str, factory: Callable[..., Any], module: Optional[str] = None) -> None:
    """Register a custom task factory under ``name``."""
    key = canonical_task_name(name)
    _USER_REGISTRY[key] = factory
    if module is not None:
        TASK_MODULES[key] = module


def canonical_task_name(name: str) -> str:
    """Map ``name`` to its canonical task name (lowercase, aliases resolved)."""
    if not isinstance(name, str):
        raise TypeError(f"task name must be a string, got {type(name)!r}")
    key = name.strip().lower().replace(" ", "_")
    key = TASK_ALIASES.get(key, key)
    return key


def available_tasks() -> List[str]:
    """Return the sorted list of registered task names."""
    names = set(TASK_MODULES) | set(_USER_REGISTRY)
    return sorted(names)


def load_task_module(name: str) -> Any:
    """Import (and cache) the module implementing ``name``."""
    key = canonical_task_name(name)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    if key not in TASK_MODULES:
        raise KeyError(
            f"unknown task {name!r}; available tasks: {', '.join(available_tasks())}"
        )
    module = importlib.import_module(TASK_MODULES[key])
    _MODULE_CACHE[key] = module
    return module


def task_class(name: str, *, required: bool = True) -> Optional[Type[Any]]:
    """Return the task class implementing ``name`` (``None`` when unavailable)."""
    key = canonical_task_name(name)
    if key in _USER_REGISTRY:
        return None
    try:
        module = load_task_module(key)
    except ImportError:
        if required:
            raise
        return None
    for attr in ("Task", "Simulator", "SimulationTask"):
        cls = getattr(module, attr, None)
        if isinstance(cls, type):
            return cls
    # fall back to the first class defined in the module that exposes prior_sample
    for value in vars(module).values():
        if isinstance(value, type) and hasattr(value, "prior_sample"):
            return value
    if required:
        raise AttributeError(
            f"module {TASK_MODULES.get(key)!r} does not define a task class"
        )
    return None


def build_task(name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate task ``name``.

    If the module exposes a ``build_task`` factory it is used, otherwise the
    task class is constructed directly.  Extra keyword arguments are forwarded
    to the constructor (or to the factory's ``**kwargs``).
    """
    key = canonical_task_name(name)
    if key in _USER_REGISTRY:
        return _USER_REGISTRY[key](*args, **kwargs)
    module = load_task_module(key)
    factory = getattr(module, "build_task", None)
    if callable(factory):
        return factory(*args, **kwargs)
    cls = task_class(key)
    assert cls is not None
    return cls(*args, **kwargs)


def get_task(name: str, *args: Any, **kwargs: Any) -> Any:
    """Alias of :func:`build_task` (paper scripts use both names)."""
    return build_task(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# Shared task interface
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Light-weight description of a task's parameter/data layout."""

    name: str
    n_parameters: int
    n_data: int
    parameter_names: Sequence[str] = ()
    data_names: Sequence[str] = ()
    metadata_dim: int = 0
    function_valued: Tuple[Any, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.parameter_names:
            self.parameter_names = tuple(f"theta_{i}" for i in range(self.n_parameters))
        if not self.data_names:
            self.data_names = tuple(f"x_{i}" for i in range(self.n_data))

    @property
    def n_variables(self) -> int:
        return self.n_parameters + self.n_data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_parameters": self.n_parameters,
            "n_data": self.n_data,
            "parameter_names": list(self.parameter_names),
            "data_names": list(self.data_names),
            "metadata_dim": self.metadata_dim,
        }


class TaskBase:
    """Common behaviour shared by all Simformer simulators.

    Sub-classes implement :meth:`prior_sample` (and usually
    :meth:`log_prior` and :meth:`log_likelihood`); the joint-vector helpers,
    dataset generation, condition masks and model factories are provided here
    so that every task plugs directly into ``training.py``/``sampling.py``.

    Conventions (Source: Sec. 3.1 and the Addendum "Tokenization"):

    * the joint vector is ``[theta_1 .. theta_P, x_1 .. x_D]`` where ``x`` is
      flattened in row-major order when the data is a time series of several
      observed series;
    * ``condition_mask[i] = 1`` marks an *observed* variable (kept clean),
      ``condition_mask[i] = 0`` marks a latent variable (modelled by the score
      network).
    """

    #: overridden by sub-classes
    name: str = "task"
    n_parameters: int = 0
    n_data: int = 0
    parameter_names: Sequence[str] = ()
    data_names: Sequence[str] = ()

    # ------------------------------------------------------------------
    # data generation
    # ------------------------------------------------------------------
    def prior_sample(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw ``n_samples`` parameter vectors, shape ``(n_samples, n_parameters)``."""
        raise NotImplementedError

    def simulate(
        self,
        theta: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        n_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Simulate observations ``x ~ p(x|theta)``, shape ``(n, n_data)``."""
        raise NotImplementedError

    def __call__(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate ``n_samples`` joint samples; returns ``(theta, x)``."""
        rng = _check_rng(rng)
        theta = self.prior_sample(n_samples, rng)
        x = self.simulate(theta, rng, **kwargs)
        return theta, x

    def make_dataset(
        self,
        n_simulations: int,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        chunk_size: int = 1024,
        verbose: bool = False,
    ) -> np.ndarray:
        """Generate a simulation budget as a joint array ``(N, P + D)``.

        Source: Sec. 4.1 (budgets of 1k / 10k / 100k simulations).
        """
        rng = _check_rng(rng, seed=seed)
        outs: List[np.ndarray] = []
        remaining = int(n_simulations)
        chunk = max(1, int(chunk_size))
        while remaining > 0:
            m = min(chunk, remaining)
            theta, x = self(m, rng)
            outs.append(self.to_joint(theta, x))
            remaining -= m
            if verbose:
                print(f"[{self.name}] simulated {int(n_simulations) - remaining}/{int(n_simulations)}")
        return np.concatenate(outs, axis=0)

    # ------------------------------------------------------------------
    # joint vector helpers
    # ------------------------------------------------------------------
    @property
    def joint_dim(self) -> int:
        return int(self.n_parameters + self.n_data)

    def to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Concatenate ``(theta, x)`` into the joint vector."""
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if theta.shape[1] != self.n_parameters and theta.shape[0] == self.n_parameters:
            theta = theta.T
        if x.shape[1] != self.n_data and x.shape[0] == self.n_data:
            x = x.T
        if theta.shape[0] != x.shape[0]:
            if theta.shape[0] == 1:
                theta = np.repeat(theta, x.shape[0], axis=0)
            elif x.shape[0] == 1:
                x = np.repeat(x, theta.shape[0], axis=0)
            else:
                raise ValueError(
                    f"cannot align theta {theta.shape} with x {x.shape} for task {self.name}"
                )
        return np.concatenate([theta, x], axis=1)

    def split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split a joint array into ``(theta, x)``."""
        joint = np.atleast_2d(np.asarray(joint, dtype=np.float64))
        p = int(self.n_parameters)
        return joint[:, :p], joint[:, p:]

    # ------------------------------------------------------------------
    # condition masks
    # ------------------------------------------------------------------
    def posterior_condition_mask(self, n_parameters: Optional[int] = None, n_data: Optional[int] = None) -> np.ndarray:
        """``M_C`` for the posterior ``p(theta|x)``: parameters latent, data observed."""
        p = self.n_parameters if n_parameters is None else int(n_parameters)
        d = self.n_data if n_data is None else int(n_data)
        return np.concatenate([np.zeros(p), np.ones(d)]).astype(np.float32)

    def likelihood_condition_mask(self, n_parameters: Optional[int] = None, n_data: Optional[int] = None) -> np.ndarray:
        """``M_C`` for the likelihood ``p(x|theta)``: parameters observed, data latent."""
        p = self.n_parameters if n_parameters is None else int(n_parameters)
        d = self.n_data if n_data is None else int(n_data)
        return np.concatenate([np.ones(p), np.zeros(d)]).astype(np.float32)

    def joint_condition_mask(self) -> np.ndarray:
        """``M_C`` for the joint density ``p(theta, x)`` (nothing observed)."""
        return np.zeros(self.joint_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # model factories
    # ------------------------------------------------------------------
    def token_spec(self, token_dim: int = 50, **kwargs: Any) -> Any:
        """Build the :class:`~simformer.tokenizer.TokenSpec` for this task."""
        from simformer.tokenizer import TokenSpec  # local import (optional dep)

        return TokenSpec(
            parameter_names=tuple(self.parameter_names),
            data_names=tuple(self.data_names),
        )

    def build_tokenizer(self, token_dim: int = 50, **kwargs: Any) -> Any:
        from simformer.tokenizer import Tokenizer

        return Tokenizer(self.token_spec(token_dim=token_dim, **kwargs), token_dim=token_dim)

    def attention_mask(self, **kwargs: Any) -> np.ndarray:
        """Task-specific directed attention mask ``M_E`` (Sec. 3.2)."""
        from simformer.attention_masks import build_attention_mask

        return build_attention_mask(
            self.name,
            n_theta=self.n_parameters,
            n_x=self.n_data,
            **kwargs,
        )

    def build_model(self, **kwargs: Any) -> Any:
        """Build a Simformer score network for this task."""
        from simformer.transformer import build_score_network

        return build_score_network(task=self.name, spec=self.token_spec(), **kwargs)

    build_score_network = build_model

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def spec(self) -> TaskSpec:
        return TaskSpec(
            name=self.name,
            n_parameters=self.n_parameters,
            n_data=self.n_data,
            parameter_names=tuple(self.parameter_names),
            data_names=tuple(self.data_names),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.spec().to_dict()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, n_parameters={self.n_parameters}, n_data={self.n_data})"


# ---------------------------------------------------------------------------
# helpers shared with the task sub-modules
# ---------------------------------------------------------------------------


def _check_rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    """Return ``rng`` unchanged, or a fresh generator (optionally seeded)."""
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def task_dimensions(name: str) -> Tuple[int, int]:
    """Return ``(n_parameters, n_data)`` for task ``name``."""
    task = build_task(name)
    p = int(getattr(task, "n_parameters", len(getattr(task, "parameter_names", ()))))
    d = int(getattr(task, "n_data", len(getattr(task, "data_names", ()))))
    return p, d


def task_attention_mask(name: str, **kwargs: Any) -> np.ndarray:
    """Task attention mask without instantiating the simulator."""
    from simformer.attention_masks import build_attention_mask

    return build_attention_mask(name, **kwargs)


def attach_attention_mask(model: Any, task: Any = None, mask: Any = None, **kwargs: Any) -> Any:
    """Attach an attention mask to ``model`` (returns the model).

    Several network classes in this code base accept ``attention_mask`` either
    as a constructor argument or as an attribute; both are supported.
    """
    if mask is None:
        if task is None:
            raise ValueError("either `task` or `mask` must be provided")
        mask = task.attention_mask(**kwargs) if isinstance(task, TaskBase) else task
    if hasattr(model, "attention_mask"):
        try:
            model.attention_mask = mask
        except Exception:  # pragma: no cover - attribute may be read-only
            pass
    if hasattr(model, "set_attention_mask"):
        try:
            model.set_attention_mask(mask)
        except Exception:  # pragma: no cover
            pass
    return model


# ---------------------------------------------------------------------------
# lazy attribute access: ``simformer.tasks.two_moons``
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:  # pragma: no cover - import sugar
    key = canonical_task_name(name)
    if key in TASK_MODULES or key in _USER_REGISTRY:
        return load_task_module(key)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - introspection sugar
    return sorted(set(globals()) | set(TASK_MODULES))


# ---------------------------------------------------------------------------
# names of the benchmark tasks (Sec. 4.1) and scientific tasks (Sec. 4.2-4.4)
# ---------------------------------------------------------------------------

BENCHMARK_TASKS: Tuple[str, ...] = (
    "gaussian_linear",
    "gaussian_mixture",
    "two_moons",
    "slcp",
)

ARBITRARY_CONDITIONAL_TASKS: Tuple[str, ...] = BENCHMARK_TASKS + ("tree", "hmm")

SCIENTIFIC_TASKS: Tuple[str, ...] = ("lotka_volterra", "sird", "hodgkin_huxley")

__all__ += [
    "BENCHMARK_TASKS",
    "ARBITRARY_CONDITIONAL_TASKS",
    "SCIENTIFIC_TASKS",
    "canonical_task_name",
]
