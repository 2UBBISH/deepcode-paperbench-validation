#!/usr/bin/env python
"""Train the RICE mask network (Algorithm 1) for every in-scope environment.

This script is the driver for CORE COMPONENT #1 / Experiment I's efficiency half
(Table 4).  For each task it

1. loads the warm-start (bottlenecked) policy ``pi`` produced by
   ``scripts/pretrain_agent.py`` (or an untrained policy, for smoke tests),
2. builds the re-designed *StateMask* mask network ``pi_tilde_theta`` whose
   architecture mirrors the target agent (Appendix C.2 / addendum), and
3. runs Algorithm 1 for a **fixed number of environment samples** taken from
   Table 4 of the paper (``3e5`` for the dense MuJoCo games, ``1.5e6`` for
   Selfish Mining, ``1e7`` for CAGE Challenge 2, ``2443260`` for MetaDrive
   Macro-v1), recording wall-clock seconds so that the ``~16.8%`` faster
   mask-training claim can be trend-checked against the released StateMask.

Paper references (verbatim)
---------------------------
* Masking rule, Eq. (1)::

      a_t (.) a_t^m = a_t                 if a_t^m = 0
                      a_random            if a_t^m = 1

* Objective, Eq. (2) / Theorem 3.3::

      J(theta) = min |eta(pi) - eta(pi_bar)|   -->   J(theta) = max eta(pi_bar)

* Blinding bonus::

      R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m

* Importance score = "the probability of mask network outputting '0'" (§3.3).
* ``alpha = 0.0001`` (Table 3; note that the §C.3 text says ``0.01`` — Table 3
  is operative per the reproduction addendum).

The script is deliberately tolerant: it boots from any of the three repository
layouts used during development, imports the environment/baseline modules lazily
and degrades gracefully (with a recorded note) when a component is missing.

Usage
-----
    python rice/scripts/train_mask.py --task Hopper-v3 --seeds 0 1 2
    python rice/scripts/train_mask.py --all
    python rice/scripts/train_mask.py --task Reacher-v2 --samples 1e4 --plot

Outputs (under ``--out-dir``)::

    weights/{task}_seed{seed}.pt   trained mask network state dicts
    logs/{task}_seed{seed}.json    per-run training log / timing
    mask_{task}.json               aggregated (mean/std) results
    mask_{task}.csv                CSV table (samples, seconds, ratio)
    table4_{task}.png              Fig. 5 / Table 4 style timing plot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (repo root == outer ``rice/``, inner ``rice/rice`` or installed)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    _HERE,
    os.path.dirname(_HERE),
    os.path.dirname(os.path.dirname(_HERE)),
):
    if _cand and _cand not in sys.path:
        sys.path.insert(0, _cand)


# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------
#: Table 4 -- "Num. of samples" used to train the mask network.
MASK_SAMPLE_BUDGETS: Dict[str, float] = {
    "Hopper-v3": 3e5,
    "Walker2d-v3": 3e5,
    "Reacher-v2": 3e5,
    "HalfCheetah-v3": 3e5,
    "SparseHopper": 3e5,
    "SparseWalker2d": 3e5,      # out of scope (kept to avoid KeyError)
    "SparseHalfCheetah": 3e5,
    "SelfishMining": 1.5e6,
    "CageChallenge2": 1e7,
    "Macro-v1": 2443260,
    "MalwareMutation": 32349,   # out of scope (Table 7 is excluded)
}

#: Table 4 -- wall-clock seconds reported for the released StateMask baseline.
TABLE4_STATEMASK_SECONDS: Dict[str, float] = {
    "Hopper-v3": 15393.0,
    "Walker2d-v3": 2240.0,
    "Reacher-v2": 8571.0,
    "HalfCheetah-v3": 1579.0,
    "SelfishMining": 9520.0,
    "CageChallenge2": 79382.0,
    "Macro-v1": 109802.0,
    "MalwareMutation": 50775.0,
}

#: Table 4 -- wall-clock seconds reported for *our* mask network (trend target).
TABLE4_OURS_SECONDS: Dict[str, float] = {
    "Hopper-v3": 12426.0,
    "Walker2d-v3": 1899.0,
    "Reacher-v2": 7033.0,
    "HalfCheetah-v3": 1317.0,
    "SelfishMining": 8360.0,
    "CageChallenge2": 65400.0,
    "Macro-v1": 88761.0,
    "MalwareMutation": 41340.0,
}

#: Paper's reported average mask-training time reduction (§C.3 Table 4).
EFFICIENCY_DROP_REFERENCE: float = 0.168

#: Table 3 -- ``alpha`` (mask-network blinding bonus coefficient).
#: Note: §C.3 text states 0.01, but Table 3 is operative (per the addendum).
ALPHA_BY_TASK: Dict[str, float] = {
    "Hopper-v3": 0.0001,
    "Walker2d-v3": 0.0001,
    "Reacher-v2": 0.0001,
    "HalfCheetah-v3": 0.0001,
    "SparseHopper": 0.0001,
    "SparseWalker2d": 0.0001,
    "SparseHalfCheetah": 0.0001,
    "SelfishMining": 0.0001,
    "CageChallenge2": 0.0001,
    "Macro-v1": 0.0001,
    "MalwareMutation": 0.0001,
}
DEFAULT_ALPHA: float = 0.0001

#: Sweep grid for Experiment V (α sensitivity, Figure 9).
ALPHA_SWEEP_VALUES: Tuple[float, ...] = (0.01, 0.001, 0.0001)

DENSE_TASKS: Tuple[str, ...] = (
    "Hopper-v3",
    "Walker2d-v3",
    "Reacher-v2",
    "HalfCheetah-v3",
    "SelfishMining",
    "CageChallenge2",
    "Macro-v1",
)
SPARSE_TASKS: Tuple[str, ...] = ("SparseHopper", "SparseHalfCheetah")
ALL_TASKS: Tuple[str, ...] = DENSE_TASKS + SPARSE_TASKS
OUT_OF_SCOPE_TASKS: Tuple[str, ...] = ("SparseWalker2d", "MalwareMutation")

TASK_ALIASES: Dict[str, str] = {
    "hopper": "Hopper-v3",
    "hopperv3": "Hopper-v3",
    "hopper-v3": "Hopper-v3",
    "walker": "Walker2d-v3",
    "walker2d": "Walker2d-v3",
    "walker2d-v3": "Walker2d-v3",
    "reacher": "Reacher-v2",
    "reacher-v2": "Reacher-v2",
    "halfcheetah": "HalfCheetah-v3",
    "halfcheetah-v3": "HalfCheetah-v3",
    "sparsehopper": "SparseHopper",
    "sparse-walker2d": "SparseWalker2d",
    "sparsewalker2d": "SparseWalker2d",
    "sparsehalfcheetah": "SparseHalfCheetah",
    "selfish": "SelfishMining",
    "selfishmining": "SelfishMining",
    "selfish-mining": "SelfishMining",
    "cage": "CageChallenge2",
    "cagechallenge2": "CageChallenge2",
    "cage-challenge-2": "CageChallenge2",
    "auto": "Macro-v1",
    "macro": "Macro-v1",
    "macro-v1": "Macro-v1",
    "autodriving": "Macro-v1",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def canonical_task(name: str) -> str:
    """Normalise a task spelling/alias to the canonical RICE task name."""
    if not name:
        return "Hopper-v3"
    key = str(name).strip().lower().replace("_", "-").replace(" ", "")
    if name in ALL_TASKS or name in OUT_OF_SCOPE_TASKS:
        return name
    key2 = key.replace("-", "")
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    if key2 in TASK_ALIASES:
        return TASK_ALIASES[key2]
    for task in ALL_TASKS + OUT_OF_SCOPE_TASKS:
        if task.lower().replace("-", "") == key2:
            return task
    return name


def is_sparse(task: str) -> bool:
    """Whether ``task`` is a sparse-reward (Experiment II) variant."""
    return canonical_task(task) in SPARSE_TASKS


def task_budget(task: str, override: Optional[float] = None) -> float:
    """Table 4 sample budget (number of environment samples) for ``task``."""
    if override is not None:
        return float(override)
    return float(MASK_SAMPLE_BUDGETS.get(canonical_task(task), 3e5))


def task_alpha(task: str, override: Optional[float] = None) -> float:
    """Table 3 ``alpha`` (blinding-bonus coefficient) for ``task``."""
    if override is not None:
        return float(override)
    return float(ALPHA_BY_TASK.get(canonical_task(task), DEFAULT_ALPHA))


def resolve_device(device: str = "auto") -> str:
    """Resolve ``auto`` -> ``cuda``/``cpu``."""
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch optional
        return "cpu"


def import_first(module_names: Sequence[str]) -> Optional[Any]:
    """Import the first importable dotted module from ``module_names``."""
    import importlib

    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first existing attribute from ``names`` on ``obj``."""
    if obj is None:
        return default
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                continue
    return default


def ensure_dir(path: str) -> str:
    """Create ``path`` if needed and return it."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy/torch scalars and arrays."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:  # torch tensors
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class MaskTrainingConfig:
    """Settings for one Algorithm-1 mask-training run.

    All fields mirror either Table 3/Table 4 of the paper or a documented
    default for an unspecified hyper-parameter (PPO settings fall back to the
    Stable-Baselines3 defaults used by :class:`rice.algorithms.ppo.PPOConfig`).
    """

    task: str = "Hopper-v3"
    #: Table 4 sample budget (environment steps spent training the mask).
    samples: Optional[float] = None
    #: Table 3 blinding-bonus coefficient.
    alpha: Optional[float] = None
    #: Mask-network architecture -- ``None`` -> per-environment default.
    net_arch: Optional[Tuple[int, ...]] = None
    activation: str = "tanh"
    #: Algorithm-1 outer iterations when ``samples`` should be split (default
    #: derived from ``samples // max_episode_steps`` with a sane lower bound).
    n_iterations: Optional[int] = None
    #: Steps collected per outer iteration (``None`` -> one episode length T).
    steps_per_iter: Optional[int] = None
    #: Exploration-policy settings (SB3 PPO defaults via PPOConfig).
    learning_rate: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    #: Seeds.
    n_seeds: int = 3
    seeds: Optional[List[int]] = None
    seed: int = 0
    #: Bookkeeping.
    device: str = "auto"
    out_dir: str = "results/mask"
    weights: Optional[str] = None       # warm-start target policy checkpoint
    verbose: int = 1
    log_every: int = 10
    #: Extra env kwargs forwarded to ``make_env``.
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    def seed_list(self) -> List[int]:
        """Explicit seed list (``[seeds]`` if provided, else ``range(n_seeds)``)."""
        if self.seeds:
            return [int(s) for s in self.seeds]
        return [int(self.seed) + i for i in range(max(1, int(self.n_seeds)))]

    def resolved_samples(self) -> float:
        """Table 4 budget for the task (or the explicit override)."""
        return task_budget(self.task, self.samples)

    def resolved_alpha(self) -> float:
        """Table 3 ``alpha`` for the task (or the explicit override)."""
        return task_alpha(self.task, self.alpha)

    def clone(self, **overrides: Any) -> "MaskTrainingConfig":
        """Copy of the config with ``overrides`` applied."""
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if k in data or k in self.__dataclass_fields__})
        extra = {k: v for k, v in overrides.items() if k not in data and k not in self.__dataclass_fields__}
        cfg = MaskTrainingConfig(**data)
        for k, v in extra.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any) -> "MaskTrainingConfig":
        """Build a config from a (YAML-parsed) mapping plus kwargs."""
        data: Dict[str, Any] = dict(mapping or {})
        # tolerate a few aliases used by the configs/ layer
        if "bonus" in data and "alpha" not in data:
            data["alpha"] = data.pop("bonus")
        if "num_samples" in data and "samples" not in data:
            data["samples"] = data.pop("num_samples")
        if "total_samples" in data and "samples" not in data:
            data["samples"] = data.pop("total_samples")
        if "iterations" in data and "n_iterations" not in data:
            data["n_iterations"] = data.pop("iterations")
        if "env_kwargs" in data and isinstance(data["env_kwargs"], dict):
            pass
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        unknown = {k: v for k, v in data.items() if k not in known}
        clean.update({k: v for k, v in overrides.items() if k in known})
        cfg = cls(**clean)
        for k, v in {**unknown, **{k: v for k, v in overrides.items() if k not in known}}.items():
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view of the config (with resolved values)."""
        data = asdict(self)
        data["samples"] = self.resolved_samples()
        data["alpha"] = self.resolved_alpha()
        data["seeds"] = self.seed_list()
        return data


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
class MaskTrainer:
    """Trains the RICE mask network (Algorithm 1) for a single task.

    Responsibilities:

    * construct the environment and the warm-start target policy,
    * build the mask network with the *target agent's* architecture
      (MuJoCo ``(64, 64)``, Selfish Mining ``(128,128,128,128)``,
      CAGE-2 ``(64, 64, 64)``, Macro-v1 DI-engine VAC default),
    * run Algorithm 1 for the Table 4 sample budget and time it,
    * save weights so ``scripts/run_refine.py`` can reuse the explanation.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = self._config_from_args(args)
        self.notes: List[str] = []
        self.device = resolve_device(self.config.device)
        self.task = canonical_task(self.config.task)
        self.logs: List[Dict[str, Any]] = []

    # -- config ------------------------------------------------------------
    @staticmethod
    def _config_from_args(args: argparse.Namespace) -> MaskTrainingConfig:
        cfg = MaskTrainingConfig.from_mapping(getattr(args, "config", None))
        overrides: Dict[str, Any] = {}
        for key in (
            "task",
            "samples",
            "alpha",
            "net_arch",
            "activation",
            "n_iterations",
            "steps_per_iter",
            "learning_rate",
            "n_epochs",
            "batch_size",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "seed",
            "device",
            "out_dir",
            "weights",
            "verbose",
            "log_every",
        ):
            value = getattr(args, key, None)
            if value is not None:
                overrides[key] = value
        if getattr(args, "seeds", None):
            overrides["seeds"] = [int(s) for s in args.seeds]
        if getattr(args, "n_seeds", None) is not None:
            overrides["n_seeds"] = int(args.n_seeds)
        env_kwargs = getattr(args, "env_kwargs", None)
        if env_kwargs:
            overrides["env_kwargs"] = dict(env_kwargs)
        cfg = cfg.clone(**overrides) if overrides else cfg
        cfg.task = canonical_task(cfg.task)
        return cfg

    # -- module access -----------------------------------------------------
    def _import(self, candidates: Sequence[str]) -> Optional[Any]:
        module = import_first(candidates)
        if module is None:
            self.notes.append(f"could not import any of {list(candidates)}")
        return module

    def mask_module(self) -> Optional[Any]:
        """The ``rice.algorithms.mask_network`` module."""
        return self._import(
            (
                "rice.algorithms.mask_network",
                "rice.rice.algorithms.mask_network",
                "algorithms.mask_network",
            )
        )

    def ppo_module(self) -> Optional[Any]:
        """The ``rice.algorithms.ppo`` module."""
        return self._import(
            (
                "rice.algorithms.ppo",
                "rice.rice.algorithms.ppo",
                "algorithms.ppo",
            )
        )

    def refine_module(self) -> Optional[Any]:
        """The ``rice.algorithms.refine`` module (weight loading / evaluation)."""
        return self._import(
            (
                "rice.algorithms.refine",
                "rice.rice.algorithms.refine",
                "algorithms.refine",
            )
        )

    # -- construction ------------------------------------------------------
    def build_env(self, seed: Optional[int] = None) -> Any:
        """Environment factory using the RICE registry with per-module fallback."""
        make_env = None
        env_mod = self._import(
            (
                "rice.environments",
                "rice.rice.environments",
                "environments",
            )
        )
        if env_mod is not None:
            make_env = _get(env_mod, "make_env")
        if make_env is None:
            module_name = {
                "SparseHopper": "mujoco_sparse",
                "SparseWalker2d": "mujoco_sparse",
                "SparseHalfCheetah": "mujoco_sparse",
                "SelfishMining": "selfish_mining",
                "CageChallenge2": "cage_challenge2",
                "Macro-v1": "autodriving",
            }.get(self.task, "mujoco_dense")
            mod = self._import(
                (
                    f"rice.environments.{module_name}",
                    f"rice.rice.environments.{module_name}",
                    f"environments.{module_name}",
                    module_name,
                )
            )
            if mod is not None:
                make_env = _get(mod, "make_env")
        if make_env is None:
            raise ImportError(
                "no environment factory found; install the rice.environments package "
                "or run from the repository root"
            )
        kwargs = dict(self.config.env_kwargs or {})
        if seed is not None:
            kwargs.setdefault("seed", int(seed))
        try:
            return make_env(self.task, **kwargs)
        except TypeError:
            return make_env(**kwargs)

    def net_arch(self) -> Tuple[int, ...]:
        """Mask/target architecture for the task (§C.2 / addendum)."""
        if self.config.net_arch:
            return tuple(int(x) for x in self.config.net_arch)
        env_mod = self._import(
            (
                "rice.environments",
                "rice.rice.environments",
                "environments",
            )
        )
        arch = _get(env_mod, "default_net_arch") if env_mod is not None else None
        if callable(arch):
            try:
                return tuple(int(x) for x in arch(self.task))
            except Exception:
                pass
        # Documented fallback (matches environments/__init__.py).
        defaults = {
            "SelfishMining": (128, 128, 128, 128),
            "CageChallenge2": (64, 64, 64),
            "Macro-v1": (256, 256),
        }
        return tuple(defaults.get(self.task, (64, 64)))

    def build_policy(self, env: Any, seed: Optional[int] = None) -> Any:
        """Warm-start target policy ``pi`` (weights from ``--weights`` if given).

        Preference order: an explicit checkpoint (``--weights``), then an
        ``ActorCritic`` from ``rice.algorithms.ppo``.  A randomly initialised
        policy is acceptable for smoke-testing the mask-training loop but the
        paper's explanation assumes a *bottlenecked but reasonable* policy
        (Assumption 3.2), so pre-training first is strongly recommended.
        """
        ppo = self.ppo_module()
        policy = None
        if ppo is not None:
            actor_critic = _get(ppo, "ActorCritic")
            if actor_critic is not None:
                try:
                    policy = actor_critic(
                        env.observation_space,
                        env.action_space,
                        net_arch=self.net_arch(),
                        activation=self.config.activation,
                        device=self.device,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    self.notes.append(f"ActorCritic construction failed: {exc}")
                    policy = None
        if policy is None:
            policy = _get(env.action_space, "sample")  # callable fallback
            self.notes.append("no torch policy available; using action-space sampler")

        path = self.config.weights
        if path and os.path.exists(path):
            ref = self.refine_module()
            loader = _get(ref, "load_policy_weights") if ref is not None else None
            if callable(loader) and policy is not None:
                try:
                    loader(policy, path)
                    self.notes.append(f"warm-started target policy from {path}")
                except Exception as exc:
                    self.notes.append(f"failed to load warm-start weights ({exc})")
        elif path:
            self.notes.append(f"weights file not found: {path}")

        if seed is not None:
            try:
                import torch

                torch.manual_seed(int(seed))
            except Exception:
                pass
        return policy

    def build_mask_network(self, env: Any, seed: Optional[int] = None) -> Any:
        """Instantiate the re-designed StateMask mask network."""
        mask_mod = self.mask_module()
        if mask_mod is None:
            raise ImportError("rice.algorithms.mask_network is required to train a mask")
        mask_cls = _get(mask_mod, "MaskNetwork")
        if mask_cls is None:
            raise ImportError("rice.algorithms.mask_network.MaskNetwork not found")
        return mask_cls(
            observation_space=env.observation_space,
            net_arch=self.net_arch(),
            activation=self.config.activation,
            device=self.device,
        )

    def build_config(self) -> Any:
        """Build the ``MaskNetworkConfig`` honoring the Table 4 sample budget."""
        mask_mod = self.mask_module()
        cfg_cls = _get(mask_mod, "MaskNetworkConfig") if mask_mod is not None else None
        samples = float(self.config.resolved_samples())
        alpha = float(self.config.resolved_alpha())
        if cfg_cls is None:
            return None

        # Derive the outer-iteration count from the sample budget when the user
        # did not pin one explicitly: each iteration spends T environment steps.
        n_iterations = self.config.n_iterations
        if n_iterations is None:
            try:
                steps = int(self.config.steps_per_iter or 0)
            except Exception:
                steps = 0
            if steps <= 0:
                steps = 1000
            n_iterations = max(1, int(round(samples / float(steps))))

        kwargs: Dict[str, Any] = {
            "alpha": alpha,
            "net_arch": self.net_arch(),
            "activation": self.config.activation,
            "n_iterations": int(n_iterations),
            "total_samples": samples,
            "device": self.device,
            "verbose": int(self.config.verbose),
            "log_every": int(self.config.log_every),
        }
        if self.config.steps_per_iter:
            kwargs["max_steps_per_iter"] = int(self.config.steps_per_iter)
        # Forward PPO settings through PPOConfig when the trainer accepts it.
        ppo = self.ppo_module()
        ppo_cfg_cls = _get(ppo, "PPOConfig") if ppo is not None else None
        if ppo_cfg_cls is not None:
            try:
                ppo_cfg = ppo_cfg_cls(
                    learning_rate=float(self.config.learning_rate),
                    n_epochs=int(self.config.n_epochs),
                    batch_size=int(self.config.batch_size),
                    gamma=float(self.config.gamma),
                    gae_lambda=float(self.config.gae_lambda),
                    clip_range=float(self.config.clip_range),
                    ent_coef=float(self.config.ent_coef),
                    vf_coef=float(self.config.vf_coef),
                    max_grad_norm=float(self.config.max_grad_norm),
                    net_arch=tuple(self.net_arch()),
                    activation=self.config.activation,
                    device=self.device,
                )
                kwargs["policy_config"] = ppo_cfg
            except Exception:
                pass

        # Drop fields the dataclass does not declare (naming drift tolerance).
        known = set(getattr(cfg_cls, "__dataclass_fields__", {}) or {})
        if known:
            dropped = [k for k in kwargs if k not in known]
            if dropped:
                self.notes.append(f"MaskNetworkConfig ignored unknown fields: {dropped}")
            kwargs = {k: v for k, v in kwargs.items() if k in known}
            if "n_iterations" not in known and "total_samples" not in known:
                self.notes.append(
                    "MaskNetworkConfig exposes neither 'n_iterations' nor "
                    "'total_samples'; the Table 4 budget may not be honored exactly"
                )
        try:
            return cfg_cls(**kwargs)
        except TypeError as exc:  # fall back to a permissive positional config
            self.notes.append(f"MaskNetworkConfig kwargs rejected ({exc}); using defaults")
            try:
                return cfg_cls()
            except Exception:
                return None

    # -- training ----------------------------------------------------------
    def train_seed(self, seed: int) -> Dict[str, Any]:
        """Run Algorithm 1 once with ``seed`` and return a log record."""
        from rice.utils.seeding import RNG  # type: ignore

        env = self.build_env(seed=seed)
        try:
            policy = self.build_policy(env, seed=seed)
            mask_network = self.build_mask_network(env, seed=seed)
            config = self.build_config()
            mask_mod = self.mask_module()
            trainer_cls = _get(mask_mod, "MaskNetworkTrainer")
            if trainer_cls is None:
                raise ImportError("MaskNetworkTrainer not found")

            rng = np.random.default_rng(int(seed))
            budget = float(self.config.resolved_samples())

            trainer_kwargs: Dict[str, Any] = {
                "env": env,
                "target_policy": policy,
                "mask_network": mask_network,
                "config": config,
            }
            try:
                trainer = trainer_cls(rng=RNG(seed=int(seed)), **trainer_kwargs)
            except TypeError:
                trainer = trainer_cls(**trainer_kwargs)

            started = time.time()
            result = trainer.train()
            elapsed = float(time.time() - started)
            if isinstance(result, dict):
                seconds = float(result.get("seconds", elapsed) or elapsed)
                samples = float(result.get("samples", budget) or budget)
                mean_mask_rate = float(result.get("mean_mask_rate", float("nan")))
                iterations = int(result.get("iterations", getattr(config, "n_iterations", 0) or 0))
                mask_history = list(result.get("mask_history", []) or [])
            else:  # duck-typed result object
                seconds = float(_get(result, "seconds", default=elapsed) or elapsed)
                samples = float(_get(result, "samples", "env_steps", default=budget) or budget)
                mean_mask_rate = float(_get(result, "mean_mask_rate", default=float("nan")))
                iterations = int(_get(result, "iterations", default=0) or 0)
                mask_history = list(_get(result, "mask_history", default=[]) or [])

            # Anti-collapse sanity check (§3.3 trivial-solution discussion):
            # a mask that always outputs "0" has mean_mask_rate == 0.
            collapse = bool(np.isfinite(mean_mask_rate) and mean_mask_rate <= 0.0)
            if collapse:
                self.notes.append(
                    f"seed {seed}: mask collapsed to always-keep (mean mask rate 0); "
                    "increase alpha or check the bonus reward"
                )

            weights_path = self._save_weights(mask_network, seed)

            record: Dict[str, Any] = {
                "task": self.task,
                "seed": int(seed),
                "samples_target": budget,
                "samples": samples,
                "seconds": seconds,
                "iterations": iterations,
                "mean_mask_rate": mean_mask_rate,
                "mask_history": mask_history[-200:],
                "collapsed": collapse,
                "alpha": float(self.config.resolved_alpha()),
                "net_arch": list(self.net_arch()),
                "device": self.device,
                "weights": weights_path,
                "notes": list(self.notes),
            }
            self.logs.append(record)
            return record
        finally:
            try:
                env.close()
            except Exception:
                pass

    def _save_weights(self, mask_network: Any, seed: int) -> Optional[str]:
        """Persist the trained mask network state dict and return its path."""
        out = os.path.join(self.config.out_dir, "weights")
        ensure_dir(out)
        path = os.path.join(out, f"{self.task}_seed{int(seed)}.pt")
        try:
            import torch

            state = _get(mask_network, "policy_state_dict", "state_dict")
            state = state() if callable(state) else state
            torch.save(state, path)
            return path
        except Exception as exc:
            self.notes.append(f"could not save mask weights for seed {seed} ({exc})")
            return None

    # -- aggregation / reporting -------------------------------------------
    def run(self, seeds: Optional[Iterable[int]] = None) -> List[Dict[str, Any]]:
        """Run Algorithm 1 for every seed, tolerating individual failures."""
        records: List[Dict[str, Any]] = []
        for seed in (list(seeds) if seeds is not None else self.config.seed_list()):
            try:
                records.append(self.train_seed(int(seed)))
            except Exception as exc:  # keep the sweep alive
                self.notes.append(f"seed {seed} failed: {exc}")
                if int(self.config.verbose) > 0:
                    traceback.print_exc()
                records.append(
                    {
                        "task": self.task,
                        "seed": int(seed),
                        "error": str(exc),
                        "seconds": float("nan"),
                        "samples": float(self.config.resolved_samples()),
                    }
                )
        return records

    def aggregate(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Mean/std of the timing/metrics + Table 4 efficiency comparison."""
        seconds = [float(r["seconds"]) for r in records if np.isfinite(r.get("seconds", float("nan")))]
        masks = [
            float(r["mean_mask_rate"])
            for r in records
            if np.isfinite(float(r.get("mean_mask_rate", float("nan"))))
        ]
        samples = [float(r.get("samples", self.config.resolved_samples())) for r in records]
        statemask_ref = TABLE4_STATEMASK_SECONDS.get(self.task)
        ours_ref = TABLE4_OURS_SECONDS.get(self.task)
        mean_seconds = float(np.mean(seconds)) if seconds else float("nan")
        drop = None
        if statemask_ref and np.isfinite(mean_seconds) and mean_seconds > 0:
            drop = float(1.0 - mean_seconds / float(statemask_ref))
        return {
            "task": self.task,
            "samples": float(np.mean(samples)) if samples else self.config.resolved_samples(),
            "samples_target": float(self.config.resolved_samples()),
            "seconds_mean": mean_seconds,
            "seconds_std": float(np.std(seconds)) if seconds else float("nan"),
            "seconds": seconds,
            "n_seeds": len(records),
            "mean_mask_rate": float(np.mean(masks)) if masks else float("nan"),
            "collapsed": bool(any(r.get("collapsed") for r in records)),
            "alpha": float(self.config.resolved_alpha()),
            "net_arch": list(self.net_arch()),
            "reference_ours_seconds": ours_ref,
            "reference_statemask_seconds": statemask_ref,
            "efficiency_drop_vs_statemask": drop,
            "records": list(records),
            "notes": list(self.notes),
        }

    def efficiency_check(self, aggregated: Dict[str, Any], tolerance: float = 0.05) -> Dict[str, Any]:
        """Trend-check the ``~16.8%`` faster-than-StateMask claim (Table 4).

        The reproduction scope is *trends, not exact numbers* (addendum), so we
        only require that (a) the per-task speedup is positive (or, when the
        wall-clock noise dominates, within ``tolerance`` of the reference ratio)
        and (b) the mean speedup over tasks with a reference is close to the
        paper's reported 16.8%.
        """
        drop = aggregated.get("efficiency_drop_vs_statemask")
        verdict: Dict[str, Any] = {
            "task": aggregated.get("task"),
            "seconds_mean": aggregated.get("seconds_mean"),
            "reference_statemask_seconds": aggregated.get("reference_statemask_seconds"),
            "efficiency_drop": drop,
            "reference_drop": EFFICIENCY_DROP_REFERENCE,
            "tolerance": tolerance,
        }
        if drop is None:
            verdict["status"] = "no_reference"
            verdict["passed"] = None
        else:
            verdict["status"] = "faster" if drop > 0 else "slower_or_equal"
            verdict["passed"] = bool(drop > -tolerance)
        return verdict

    # -- artefacts ---------------------------------------------------------
    def save(self, aggregated: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> str:
        """Write JSON/CSV artefacts under ``--out-dir``."""
        ensure_dir(self.config.out_dir)
        json_path = os.path.join(self.config.out_dir, f"mask_{self.task}.json")
        payload = dict(aggregated)
        payload["verdict"] = self.efficiency_check(aggregated)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=json_default)

        csv_path = os.path.join(self.config.out_dir, f"mask_{self.task}.csv")
        fields = [
            "task",
            "seed",
            "samples",
            "seconds",
            "iterations",
            "mean_mask_rate",
            "collapsed",
            "alpha",
            "net_arch",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json_default(row.get(k)) for k in fields})
        notes_path = os.path.join(self.config.out_dir, f"notes_{self.task}.txt")
        with open(notes_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.notes) + "\n")
        return json_path

    def plot(self, aggregated: Dict[str, Any], out_dir: Optional[str] = None) -> Optional[str]:
        """Table 4 / Figure 5 style timing bar chart (matplotlib optional)."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            self.notes.append("matplotlib unavailable; skipping plot")
            return None
        ours = aggregated.get("seconds_mean")
        ref = aggregated.get("reference_statemask_seconds")
        ours_ref = aggregated.get("reference_ours_seconds")
        labels = ["Ours (measured)"]
        values = [ours if ours is not None else float("nan")]
        if ours_ref:
            labels.append("Ours (paper)")
            values.append(ours_ref)
        if ref:
            labels.append("StateMask (paper)")
            values.append(ref)
        out_dir = ensure_dir(out_dir or self.config.out_dir)
        path = os.path.join(out_dir, f"table4_{self.task}.png")
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        colors = ["#1f77b4", "#4c9be8", "#d62728"][: len(values)]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("mask-training seconds")
        ax.set_title(f"{self.task}: mask training time\n(samples={aggregated.get('samples')})")
        for i, value in enumerate(values):
            if value is not None and np.isfinite(value):
                ax.text(i, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface of ``train_mask.py``."""
    parser = argparse.ArgumentParser(
        description="Train the RICE mask network (Algorithm 1) with the Table 4 sample budget.",
    )
    parser.add_argument("--task", "--env", dest="task", default=None, help="task name or alias")
    parser.add_argument("--tasks", nargs="+", default=None, help="several tasks")
    parser.add_argument("--all", action="store_true", help="run every in-scope dense task")
    parser.add_argument("--include-sparse", action="store_true", help="also run the in-scope sparse tasks")
    parser.add_argument("--samples", type=float, default=None, help="override the Table 4 sample budget")
    parser.add_argument("--alpha", type=float, default=None, help="blinding-bonus coefficient (Table 3)")
    parser.add_argument("--alpha-sweep", action="store_true", help="sweep alpha over Experiment V's grid")
    parser.add_argument("--net-arch", nargs="+", type=int, default=None, help="mask-net hidden sizes")
    parser.add_argument("--activation", default=None, choices=["tanh", "relu", "leaky_relu", "elu", "gelu"])
    parser.add_argument("--n-iterations", type=int, default=None, help="Algorithm-1 outer iterations")
    parser.add_argument("--steps-per-iter", type=int, default=None, help="env steps collected per iteration")
    parser.add_argument("--learning-rate", type=float, default=None, help="mask-net PPO learning rate")
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="explicit seed list")
    parser.add_argument("--n-seeds", type=int, default=None, help="number of seeds (default 3)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", default=None, help="warm-start target-policy checkpoint")
    parser.add_argument("--out-dir", default="results/mask")
    parser.add_argument("--plot", action="store_true", help="write a Table 4 style timing figure")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--config", default=None, help="YAML config path (optional)")
    parser.add_argument("--json", action="store_true", help="print the aggregate as JSON")
    return parser


def load_yaml_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort YAML config load (tolerates a missing PyYAML)."""
    if not path:
        return None
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    mapping = load_yaml_config(args.config)

    tasks: List[str] = []
    if args.tasks:
        tasks.extend(args.tasks)
    if args.task:
        tasks.append(args.task)
    if args.all or not tasks:
        tasks.extend(DENSE_TASKS)
        if args.include_sparse:
            tasks.extend(SPARSE_TASKS)
    tasks = [canonical_task(t) for t in tasks]

    all_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    try:
        from rice.utils.seeding import set_global_seeds  # type: ignore
    except Exception:
        set_global_seeds = None  # type: ignore

    alphas: Iterable[Optional[float]] = [args.alpha]
    if args.alpha_sweep:
        alphas = list(ALPHA_SWEEP_VALUES)

    for task in tasks:
        if task in OUT_OF_SCOPE_TASKS:
            print(f"[train_mask] skipping out-of-scope task {task}")
            continue
        for alpha in alphas:
            cfg = MaskTrainingConfig.from_mapping(mapping)
            cfg.task = task
            if args.samples is not None:
                cfg.samples = float(args.samples)
            if alpha is not None:
                cfg.alpha = float(alpha)
            if args.out_dir:
                cfg.out_dir = args.out_dir
            if args.verbose is not None:
                cfg.verbose = int(args.verbose)

            runner = MaskTrainer(
                argparse.Namespace(
                    config=None,
                    task=cfg.task,
                    samples=cfg.samples,
                    alpha=cfg.alpha,
                    net_arch=args.net_arch or cfg.net_arch,
                    activation=args.activation or cfg.activation,
                    n_iterations=args.n_iterations if args.n_iterations is not None else cfg.n_iterations,
                    steps_per_iter=args.steps_per_iter,
                    learning_rate=args.learning_rate or cfg.learning_rate,
                    n_epochs=args.n_epochs or cfg.n_epochs,
                    batch_size=args.batch_size or cfg.batch_size,
                    gamma=args.gamma or cfg.gamma,
                    gae_lambda=args.gae_lambda or cfg.gae_lambda,
                    clip_range=args.clip_range or cfg.clip_range,
                    ent_coef=args.ent_coef or cfg.ent_coef,
                    vf_coef=args.vf_coef or cfg.vf_coef,
                    max_grad_norm=args.max_grad_norm or cfg.max_grad_norm,
                    seeds=args.seeds,
                    n_seeds=args.n_seeds or cfg.n_seeds,
                    seed=args.seed if args.seed is not None else cfg.seed,
                    device=args.device or cfg.device,
                    weights=args.weights or cfg.weights,
                    out_dir=cfg.out_dir,
                    verbose=cfg.verbose,
                    log_every=cfg.log_every,
                    env_kwargs=cfg.env_kwargs,
                )
            )
            if set_global_seeds is not None:
                try:
                    set_global_seeds(int(runner.config.seed_list()[0]))
                except Exception:
                    pass

            if not args.quiet:
                print(
                    f"[train_mask] task={task} alpha={runner.config.resolved_alpha()} "
                    f"samples={runner.config.resolved_samples():.0f} "
                    f"seeds={runner.config.seed_list()} device={runner.device} "
                    f"net_arch={runner.net_arch()}"
                )

            records = runner.run()
            aggregated = runner.aggregate(records)
            aggregated["alpha"] = float(runner.config.resolved_alpha())
            verdict = runner.efficiency_check(aggregated)
            aggregated["verdict"] = verdict

            if not args.quiet:
                print(
                    f"[train_mask] {task}: seconds mean={aggregated['seconds_mean']:.1f} "
                    f"(std {aggregated['seconds_std']:.1f}), "
                    f"mean_mask_rate={aggregated['mean_mask_rate']:.3f}, "
                    f"efficiency_drop={aggregated['efficiency_drop_vs_statemask']}"
                )
                if verdict.get("passed") is False:
                    print(
                        f"[train_mask] NOTE: {task} mask training is slower than the paper's "
                        f"StateMask reference ratio (trend judged insignificant per addendum)"
                    )

            runner.save(aggregated, records)
            if args.plot:
                path = runner.plot(aggregated)
                if path and not args.quiet:
                    print(f"[train_mask] wrote {path}")

            all_rows.extend(records)
            aggregate_rows.append(aggregated)

    # ----- cross-task summary -------------------------------------------
    drops = [
        float(row["efficiency_drop_vs_statemask"])
        for row in aggregate_rows
        if row.get("efficiency_drop_vs_statemask") is not None
    ]
    summary = {
        "tasks": [row["task"] for row in aggregate_rows],
        "per_task": [
            {
                "task": row["task"],
                "samples": row["samples"],
                "seconds_mean": row["seconds_mean"],
                "seconds_std": row["seconds_std"],
                "mean_mask_rate": row["mean_mask_rate"],
                "efficiency_drop_vs_statemask": row["efficiency_drop_vs_statemask"],
                "reference_statemask_seconds": row["reference_statemask_seconds"],
                "reference_ours_seconds": row["reference_ours_seconds"],
                "verdict": row.get("verdict"),
            }
            for row in aggregate_rows
        ],
        "mean_efficiency_drop": float(np.mean(drops)) if drops else None,
        "reference_efficiency_drop": EFFICIENCY_DROP_REFERENCE,
    }
    if aggregate_rows:
        ensure_dir(args.out_dir)
        summary_path = os.path.join(args.out_dir, "mask_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=json_default)
        if not args.quiet:
            print(f"[train_mask] wrote {summary_path}")

    if args.json:
        print(json.dumps(summary, indent=2, default=json_default))
    elif not args.quiet:
        print("\n[train_mask] summary")
        print(f"{'task':<18}{'samples':>12}{'seconds':>12}{'mask_rate':>12}{'drop':>10}")
        for row in summary["per_task"]:
            drop = row["efficiency_drop_vs_statemask"]
            print(
                f"{row['task']:<18}{row['samples']:>12.0f}"
                f"{row['seconds_mean']:>12.1f}{row['mean_mask_rate']:>12.3f}"
                f"{(f'{drop:+.1%}' if drop is not None else 'n/a'):>10}"
            )
        if summary["mean_efficiency_drop"] is not None:
            print(
                f"\nmean efficiency drop vs StateMask (paper {EFFICIENCY_DROP_REFERENCE:.1%}): "
                f"{summary['mean_efficiency_drop']:+.1%}"
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
