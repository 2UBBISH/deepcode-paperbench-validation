"""Baseline agents for the FRE paper (ICML 2024).

This package bundles the comparison methods used in Table 1 / Table 4 of
"Zero-Shot Reinforcement Learning via Functional Reward Encodings":

  * :mod:`gc_iql`            -- goal-conditioned IQL (addendum: goals concatenated
                                to the observation, HER goal sampling
                                p_random=0.3 / p_geometric=0.5 / p_current=0.2)
  * :mod:`gc_bc`             -- goal-conditioned behavioral cloning
                                (MLE, log-std clamped at -5.0, geometric-only
                                future-goal sampling)
  * :mod:`opal`              -- OPAL re-implementation (Ajay et al., 2020) that
                                reuses the *same* transformer encoder as FRE, plus
                                the privileged ``OPAL-10`` evaluation protocol
                                (10 unit-Gaussian skills, best rollout kept)
  * :mod:`forward_backward`  -- Forward-Backward (Touati & Ollivier, 2021) wrapper
                                around ``facebookresearch/controllable_agent``
  * :mod:`successor_features`-- Successor Features with ICM features, also built
                                on ``controllable_agent``

Per Section 5 / Table 1, FB and SF recover their task vector from **5120**
(state, reward) pairs at evaluation time, whereas FRE and the policy-conditioned
baselines use 32.  That difference is encoded in the ``*_EVAL_SAMPLES`` constants
re-exported here.

The package uses PEP 562 module level ``__getattr__`` so that ``import
fre.baselines`` stays cheap: torch / numpy / gym are only pulled in when a
specific symbol is actually touched.  This keeps ``python fre/main.py --dry-run``
and the reporting utilities importable in a bare environment.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Tuple

__all__ = [
    # ---------------------------------------------------------------- gc_iql
    "GCIQL",
    "GoalSampler",
    "GoalBatch",
    "gc_iql_FlatTrajectoryIndex",
    "make_gc_iql",
    "train_gc_iql",
    "make_gc_policy_fn",
    "gc_iql_goal_rewards_and_dones",
    # ---------------------------------------------------------------- gc_bc
    "GCBC",
    "GeometricGoalSampler",
    "geometric_goal_indices",
    "gc_bc_FlatTrajectoryIndex",
    "gc_bc_GoalBatch",
    "gc_bc_goal_rewards_and_dones",
    "bc_log_prob_loss",
    "make_gc_bc",
    "train_gc_bc",
    "make_gc_bc_policy_fn",
    # ----------------------------------------------------------------- opal
    "SkillEncoder",
    "SkillDecoder",
    "OPALModel",
    "OPALAgent",
    "OPALTrainingLoss",
    "TrajectoryWindows",
    "sample_trajectory_batch",
    "make_opal",
    "train_opal",
    "make_opal_policy_fn",
    "make_skill_policy_fns",
    "evaluate_opal_privileged",
    "evaluate_opal_suite",
    "OPAL_NUM_SKILLS",
    "OPAL_TABLE1_REFERENCE",
    # ------------------------------------------------------ forward_backward
    "FBModel",
    "ForwardBackwardAgent",
    "FBAgent",
    "FBTrainingStats",
    "ControllableAgentUnavailable",
    "find_controllable_agent",
    "controllable_agent_available",
    "build_controllable_agent_command",
    "run_controllable_agent",
    "solve_task_vector",
    "sample_eval_reward_samples",
    "make_fb_policy_fn",
    "make_forward_backward",
    "train_forward_backward",
    "evaluate_fb_suite",
    "FB_EVAL_SAMPLES",
    "FB_TABLE1_REFERENCE",
    # ------------------------------------------------------ successor_features
    "ICMFeatureExtractor",
    "RandomFeatureExtractor",
    "SFModel",
    "SuccessorFeaturesAgent",
    "SFAgent",
    "SFTrainingStats",
    "build_sf_command",
    "run_sf_controllable_agent",
    "make_sf_policy_fn",
    "make_successor_features",
    "train_successor_features",
    "evaluate_sf_suite",
    "SF_EVAL_SAMPLES",
    "SF_TABLE1_REFERENCE",
    # -------------------------------------------------------------- registry
    "BASELINE_NAMES",
    "BASELINE_ALIASES",
    "available_baselines",
    "is_available",
    "baseline_eval_samples",
    "baseline_reference_row",
    "make_baseline",
    "train_baseline",
    "load_baseline_module",
]

# Public symbol -> (submodule, attribute).  Symbols whose plain name would clash
# between modules are exported under an explicit ``gc_iql_`` / ``gc_bc_`` prefix
# (the underlying modules keep their own unprefixed names).
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # ---------------------------------------------------------------- gc_iql
    "GCIQL": ("gc_iql", "GCIQL"),
    "GoalSampler": ("gc_iql", "GoalSampler"),
    "GoalBatch": ("gc_iql", "GoalBatch"),
    "gc_iql_FlatTrajectoryIndex": ("gc_iql", "FlatTrajectoryIndex"),
    "make_gc_iql": ("gc_iql", "make_gc_iql"),
    "train_gc_iql": ("gc_iql", "train_gc_iql"),
    "make_gc_policy_fn": ("gc_iql", "make_gc_policy_fn"),
    "gc_iql_goal_rewards_and_dones": ("gc_iql", "goal_rewards_and_dones"),
    # ---------------------------------------------------------------- gc_bc
    "GCBC": ("gc_bc", "GCBC"),
    "GeometricGoalSampler": ("gc_bc", "GeometricGoalSampler"),
    "geometric_goal_indices": ("gc_bc", "geometric_goal_indices"),
    "gc_bc_FlatTrajectoryIndex": ("gc_bc", "FlatTrajectoryIndex"),
    "gc_bc_GoalBatch": ("gc_bc", "GoalBatch"),
    "gc_bc_goal_rewards_and_dones": ("gc_bc", "goal_rewards_and_dones"),
    "bc_log_prob_loss": ("gc_bc", "bc_log_prob_loss"),
    "make_gc_bc": ("gc_bc", "make_gc_bc"),
    "train_gc_bc": ("gc_bc", "train_gc_bc"),
    "make_gc_bc_policy_fn": ("gc_bc", "make_gc_bc_policy_fn"),
    # ----------------------------------------------------------------- opal
    "SkillEncoder": ("opal", "SkillEncoder"),
    "SkillDecoder": ("opal", "SkillDecoder"),
    "OPALModel": ("opal", "OPALModel"),
    "OPALAgent": ("opal", "OPALAgent"),
    "OPALTrainingLoss": ("opal", "OPALTrainingLoss"),
    "TrajectoryWindows": ("opal", "TrajectoryWindows"),
    "sample_trajectory_batch": ("opal", "sample_trajectory_batch"),
    "make_opal": ("opal", "make_opal"),
    "train_opal": ("opal", "train_opal"),
    "make_opal_policy_fn": ("opal", "make_opal_policy_fn"),
    "make_skill_policy_fns": ("opal", "make_skill_policy_fns"),
    "evaluate_opal_privileged": ("opal", "evaluate_opal_privileged"),
    "evaluate_opal_suite": ("opal", "evaluate_opal_suite"),
    "OPAL_NUM_SKILLS": ("opal", "OPAL_NUM_SKILLS"),
    "OPAL_TABLE1_REFERENCE": ("opal", "OPAL_TABLE1_REFERENCE"),
    # ------------------------------------------------------ forward_backward
    "FBModel": ("forward_backward", "FBModel"),
    "ForwardBackwardAgent": ("forward_backward", "ForwardBackwardAgent"),
    "FBAgent": ("forward_backward", "FBAgent"),
    "FBTrainingStats": ("forward_backward", "FBTrainingStats"),
    "ControllableAgentUnavailable": ("forward_backward", "ControllableAgentUnavailable"),
    "find_controllable_agent": ("forward_backward", "find_controllable_agent"),
    "controllable_agent_available": ("forward_backward", "controllable_agent_available"),
    "build_controllable_agent_command": (
        "forward_backward",
        "build_controllable_agent_command",
    ),
    "run_controllable_agent": ("forward_backward", "run_controllable_agent"),
    "solve_task_vector": ("forward_backward", "solve_task_vector"),
    "sample_eval_reward_samples": ("forward_backward", "sample_eval_reward_samples"),
    "make_fb_policy_fn": ("forward_backward", "make_fb_policy_fn"),
    "make_forward_backward": ("forward_backward", "make_forward_backward"),
    "train_forward_backward": ("forward_backward", "train_forward_backward"),
    "evaluate_fb_suite": ("forward_backward", "evaluate_fb_suite"),
    "FB_EVAL_SAMPLES": ("forward_backward", "FB_EVAL_SAMPLES"),
    "FB_TABLE1_REFERENCE": ("forward_backward", "FB_TABLE1_REFERENCE"),
    # ------------------------------------------------------ successor_features
    "ICMFeatureExtractor": ("successor_features", "ICMFeatureExtractor"),
    "RandomFeatureExtractor": ("successor_features", "RandomFeatureExtractor"),
    "SFModel": ("successor_features", "SFModel"),
    "SuccessorFeaturesAgent": ("successor_features", "SuccessorFeaturesAgent"),
    "SFAgent": ("successor_features", "SFAgent"),
    "SFTrainingStats": ("successor_features", "SFTrainingStats"),
    "build_sf_command": ("successor_features", "build_sf_command"),
    "run_sf_controllable_agent": ("successor_features", "run_sf_controllable_agent"),
    "make_sf_policy_fn": ("successor_features", "make_sf_policy_fn"),
    "make_successor_features": ("successor_features", "make_successor_features"),
    "train_successor_features": ("successor_features", "train_successor_features"),
    "evaluate_sf_suite": ("successor_features", "evaluate_sf_suite"),
    "SF_EVAL_SAMPLES": ("successor_features", "SF_EVAL_SAMPLES"),
    "SF_TABLE1_REFERENCE": ("successor_features", "SF_TABLE1_REFERENCE"),
}

# Submodules that can also be reached through attribute access, e.g.
# ``fre.baselines.gc_iql`` (mirrors ``fre.envs``).
_SUBMODULES: Tuple[str, ...] = (
    "gc_iql",
    "gc_bc",
    "opal",
    "forward_backward",
    "successor_features",
)

# ---------------------------------------------------------------------------
# Static registry metadata (torch-free, safe to import eagerly)
# ---------------------------------------------------------------------------

#: Canonical baseline identifiers, matching ``fre/main.py``'s ``AGENTS`` tuple.
BASELINE_NAMES: Tuple[str, ...] = ("gc_iql", "gc_bc", "opal", "fb", "sf")

#: Aliases accepted on the command line / in scripts.
BASELINE_ALIASES: Dict[str, str] = {
    "gc_iql": "gc_iql",
    "gciql": "gc_iql",
    "gc-iql": "gc_iql",
    "gc_bc": "gc_bc",
    "gcbc": "gc_bc",
    "gc-bc": "gc_bc",
    "bc": "gc_bc",
    "opal": "opal",
    "opal-10": "opal",
    "fb": "fb",
    "forward_backward": "fb",
    "forward-backward": "fb",
    "sf": "sf",
    "successor_features": "sf",
    "successor-features": "sf",
}

#: (submodule, factory, trainer) per canonical baseline name.
_BASELINE_BUILDERS: Dict[str, Tuple[str, str, str]] = {
    "gc_iql": ("gc_iql", "make_gc_iql", "train_gc_iql"),
    "gc_bc": ("gc_bc", "make_gc_bc", "train_gc_bc"),
    "opal": ("opal", "make_opal", "train_opal"),
    "fb": (
        "forward_backward",
        "make_forward_backward",
        "train_forward_backward",
    ),
    "sf": (
        "successor_features",
        "make_successor_features",
        "train_successor_features",
    ),
}

#: (submodule, suite-evaluator attribute) per canonical baseline name.
_BASELINE_EVALUATORS: Dict[str, Tuple[str, str]] = {
    "opal": ("opal", "evaluate_opal_suite"),
    "fb": ("forward_backward", "evaluate_fb_suite"),
    "sf": ("successor_features", "evaluate_sf_suite"),
}


def __getattr__(name: str) -> Any:
    """Resolve a public symbol lazily (PEP 562).

    The heavy baseline modules (torch, numpy, gym, the external
    ``controllable_agent`` checkout) are only imported on first attribute
    access, and the resolved object is cached in the module globals so repeat
    lookups are free.
    """

    if name in _LAZY_ATTRS:
        submodule, attribute = _LAZY_ATTRS[name]
        module = import_module(f"{__name__}.{submodule}")
        try:
            value = getattr(module, attribute)
        except AttributeError as exc:  # pragma: no cover - guards index drift
            raise AttributeError(
                f"{__name__}: submodule {submodule!r} does not define "
                f"{attribute!r} (mapped from the public name {name!r})"
            ) from exc
        globals()[name] = value
        return value

    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


# ---------------------------------------------------------------------------
# Small torch-free helper API used by the scripts and by ``fre/main.py``
# ---------------------------------------------------------------------------


def load_baseline_module(name: str) -> Any:
    """Import and return one of the baseline submodules by (aliased) name."""

    canonical = BASELINE_ALIASES.get(str(name).lower().strip())
    if canonical is None:
        raise KeyError(
            f"unknown baseline {name!r}; expected one of {sorted(BASELINE_NAMES)}"
        )
    submodule = _BASELINE_BUILDERS[canonical][0]
    return import_module(f"{__name__}.{submodule}")


def available_baselines() -> List[str]:
    """Return the canonical baseline names supported by this package."""

    return list(BASELINE_NAMES)


def is_available(name: str) -> bool:
    """True if ``name`` (or an alias) maps to a known baseline."""

    return str(name).lower().strip() in BASELINE_ALIASES


def baseline_eval_samples(name: str) -> int:
    """Number of (state, reward) samples used at evaluation time.

    Table 1 caption: "FRE utilizes only 32 examples of (state, reward) pairs
    during evaluation, while the FB and SF methods require 5120 examples to be
    consistent with prior work."
    """

    canonical = BASELINE_ALIASES.get(str(name).lower().strip())
    if canonical in ("fb", "sf"):
        # Imported lazily so this helper stays usable without torch installed;
        # fall back to the paper's literal value.
        try:
            module = load_baseline_module(canonical)
            return int(
                getattr(
                    module,
                    "FB_EVAL_SAMPLES" if canonical == "fb" else "SF_EVAL_SAMPLES",
                    5120,
                )
            )
        except Exception:  # pragma: no cover - defensive
            return 5120
    return 32


def baseline_reference_row(name: str, row: str = None) -> Any:
    """Published Table 1 numbers for a baseline.

    ``name`` may be any accepted alias; ``row`` selects a specific task/domain
    row (e.g. ``"antmaze-all"``).  When ``row`` is omitted the whole reference
    dict for that baseline is returned (or the scalar for ``opal``'s
    ``(mean, std)`` tuples when ``row`` matches directly).
    """

    canonical = BASELINE_ALIASES.get(str(name).lower().strip())
    if canonical is None:
        raise KeyError(
            f"unknown baseline {name!r}; expected one of {sorted(BASELINE_NAMES)}"
        )

    table_attr = {
        "opal": ("opal", "OPAL_TABLE1_REFERENCE"),
        "fb": ("forward_backward", "FB_TABLE1_REFERENCE"),
        "sf": ("successor_features", "SF_TABLE1_REFERENCE"),
        "gc_bc": ("gc_bc", "GCBC_TABLE1_REFERENCE"),
    }.get(canonical)

    if table_attr is None:
        # GC-IQL numbers appear only in Table 1 (kitchen 59 +- 4); they are not
        # duplicated as a module constant, per the paper.
        if canonical == "gc_iql":
            table = {"kitchen": (59.0, 4.0)}
        else:  # pragma: no cover - registry is exhaustive
            raise KeyError(f"no reference table for baseline {canonical!r}")
    else:
        submodule, attribute = table_attr
        module = import_module(f"{__name__}.{submodule}")
        table = getattr(module, attribute, {})

    if row is None:
        return table
    if isinstance(table, dict):
        return table.get(row)
    return table


def make_baseline(name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a baseline agent via its ``make_*`` factory.

    Extra positional/keyword arguments are forwarded untouched, which lets
    callers pass either a ``Config`` (``make_baseline("opal", config, obs_dim,
    action_dim)``) or plain dimensions (``make_baseline("fb", obs_dim,
    action_dim)``).
    """

    canonical = BASELINE_ALIASES.get(str(name).lower().strip())
    if canonical is None:
        raise KeyError(
            f"unknown baseline {name!r}; expected one of {sorted(BASELINE_NAMES)}"
        )
    module = load_baseline_module(canonical)
    factory_name = _BASELINE_BUILDERS[canonical][1]
    factory = getattr(module, factory_name)
    return factory(*args, **kwargs)


def train_baseline(name: str, *args: Any, **kwargs: Any) -> Any:
    """Train a baseline agent via its ``train_*`` driver.

    Returns whatever the underlying driver returns (usually an
    ``(agent, history)`` tuple).
    """

    canonical = BASELINE_ALIASES.get(str(name).lower().strip())
    if canonical is None:
        raise KeyError(
            f"unknown baseline {name!r}; expected one of {sorted(BASELINE_NAMES)}"
        )
    module = load_baseline_module(canonical)
    trainer_name = _BASELINE_BUILDERS[canonical][2]
    trainer = getattr(module, trainer_name)
    return trainer(*args, **kwargs)
