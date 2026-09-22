"""Experiment I -- Fidelity and efficiency of the explanation method (RICE, ICML 2024).

Paper specification (Cheng et al., ICML 2024, PMLR 235)
------------------------------------------------------
*Sec. 4.2 (Experiment I)*
    "To show the equivalence of our explanation method with StateMask, we compare the
    fidelity of our method with StateMask. Given a trajectory, the explanation method first
    identifies and ranks top-K important time steps. ... we let the agent fast-forward to the
    critical step and force the target agent to take random actions. Then we follow the target
    agent's policy to complete the rest of the time steps. ... We compute the fidelity score of
    each explanation method as mentioned in StateMask across 500 trajectories. We set
    K = 10%, 20%, 30%, 40% and report the fidelity of the selected methods under each setup. We
    repeat each experiment 3 times with various random seeds and report the mean and standard
    deviation. Additionally, to show the efficiency of our design, we report the training time of
    the mask network using StateMask and our method when given a fixed number of training samples."

*Sec. 4.3*
    "We observe that the fidelity scores of StateMask and our method are comparable. ...
    We observe an average of 16.8% drop in the training time compared with StateMask."

What this driver does
---------------------
For every requested application (dense MuJoCo, sparse MuJoCo and domain applications -- the
paper reports Exp. I on all of them):

1. builds the environment and loads (or optionally pre-trains) the *frozen* target policy ``pi``;
2. trains the Stage-1 explanation for each requested method using the *same* fixed sample budget
   (Table 4 budgets / ``explanation.total_timesteps``), timing the wall clock cost:
   - ``ours``      -- Algorithm 1 (vanilla PPO + blinding bonus, ``rice.explanation.mask_trainer``)
   - ``statemask`` -- primal-dual StateMask mask training (``rice.baselines.statemask_r``)
   - ``random``    -- no mask net (uninformative importance), i.e. the Random explanation baseline
3. evaluates the fidelity score ``log(d / d_max) - log(l / L)`` over
   ``n_trajectories`` (default 500) trajectories x ``seeds`` (default 3) for
   ``K in {10%, 20%, 30%, 40%}`` using ``rice.explanation.fidelity``;
4. aggregates the results (mean +- std per method/K) into a Table-1/Table-4-style report and,
   when both mask-training times are available, computes the training-time reduction
   (paper expectation: about 16.8% in favour of ``ours``).

Usage
-----
    python -m experiments.exp1_fidelity_efficiency --config hopper
    python -m experiments.exp1_fidelity_efficiency --env hopper --quick      # smoke test
    python experiments/exp1_fidelity_efficiency.py --envs hopper,walker2d \
        --n-trajectories 500 --seeds 0,1,2 --out results/exp1

Everything is deterministic given ``--seed``/``--seeds`` so that the reported mean +- std is
reproducible.  The module is import-safe: heavy optional dependencies (torch/SB3/MuJoCo) are
only touched through the already-implemented ``rice`` modules.

Source: §4.2 (Experiment Design, Experiment I), §4.3 (Experiment Results), §4.1 (Evaluation
Metrics), Appendix C.3 (Table 4 / Figure 5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Project imports.  Everything is imported lazily/defensively so that the driver can still
# be imported (and its CLI introspected) in a minimal environment, mirroring the style of
# the rest of the package.
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - exercised through the real package
    from rice.utils.io import ensure_dir, get_config, save_json
    from rice.utils.logging import Logger, format_mean_std, get_logger
    from rice.utils.seeding import seed_from, set_seed
except Exception as _exc:  # pragma: no cover
    raise ImportError(
        "rice.utils could not be imported; run this driver from the repository root "
        f"(original error: {_exc})"
    ) from _exc

try:
    from rice.envs.make_env import (
        available_envs,
        d_max_for,
        env_backend,
        env_metadata,
        make_env,
    )
    _HAS_ENVS = True
except Exception:  # pragma: no cover
    _HAS_ENVS = False

try:
    from rice.models.policies import (
        build_policy,
        load_policy,
        normalize_env_key,
        policy_arch,
        save_policy,
    )
    _HAS_POLICIES = True
except Exception:  # pragma: no cover
    _HAS_POLICIES = False

try:
    from rice.explanation.mask_network import (
        build_mask_network,
        load_mask_network,
        save_mask_network,
    )
    _HAS_MASK_NETWORK = True
except Exception:  # pragma: no cover
    _HAS_MASK_NETWORK = False

try:
    from rice.explanation.mask_trainer import (
        DEFAULT_ALPHA,
        MaskPPOConfig,
        MaskTrainer,
        train_mask_network,
    )
    _HAS_MASK_TRAINER = True
except Exception:  # pragma: no cover
    _HAS_MASK_TRAINER = False
    DEFAULT_ALPHA = 1e-4

try:
    from rice.explanation.fidelity import (
        DEFAULT_K_VALUES,
        DEFAULT_N_TRAJECTORIES,
        DEFAULT_SEEDS,
        FidelityConfig,
        FidelityEvaluator,
        FidelityResult,
        evaluate_fidelity,
        evaluate_fidelity_multi_K,
        format_fidelity_table,
        training_time_reduction,
    )
    _HAS_FIDELITY = True
except Exception:  # pragma: no cover
    _HAS_FIDELITY = False

try:
    from rice.baselines.statemask_r import (
        DEFAULT_MASK_SAMPLES,
        samples_for,
        train_statemask_network,
    )
    _HAS_STATEMASK = True
except Exception:  # pragma: no cover
    _HAS_STATEMASK = False
    DEFAULT_MASK_SAMPLES = {}

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_ENVS: Tuple[str, ...] = (
    "hopper",
    "walker2d",
    "reacher",
    "halfcheetah",
    "selfish_mining",
    "cage2",
    "autodriving",
)

#: Explanation methods compared in Experiment I.  "random" carries no mask network and is the
#: lower-bound reference used by the fidelity metric of the paper's Figure 5 / Table 4.
EXPLANATION_METHODS: Tuple[str, ...] = ("ours", "statemask", "random")

#: Mask-training sample budgets from Appendix C.3, Table 4 (fallback if the baseline module is
#: unavailable).  These are the "fixed number of training samples" of the efficiency study.
TABLE4_SAMPLES: Dict[str, int] = {
    "hopper": 300_000,
    "walker2d": 300_000,
    "reacher": 300_000,
    "halfcheetah": 300_000,
    "sparse_hopper": 300_000,
    "sparse_halfcheetah": 300_000,
    "selfish_mining": 1_500_000,
    "cage2": 10_000_000,
    "autodriving": 2_443_260,
}

#: Table 4 reference wall-clock times (seconds) -- used only as sanity references in the report.
TABLE4_TIMES: Dict[str, Dict[str, float]] = {
    "hopper": {"ours": 12426.0, "statemask": 15393.0},
    "halfcheetah": {"ours": 1317.0, "statemask": 1579.0},
    "cage2": {"ours": 65400.0, "statemask": 79382.0},
}

#: The paper's reported average training-time drop of the mask network vs. StateMask (Sec. 4.3).
PAPER_TIME_REDUCTION = 0.168

#: Table 1 reference "no refine" returns per application -- trend validation only.
REFERENCE_NO_REFINE: Dict[str, Optional[float]] = {
    "hopper": 3559.44,
    "walker2d": 3339.68,
    "halfcheetah": 4540.50,
    "selfish_mining": None,
    "cage2": -23.64,
    "autodriving": 10.30,
    "reacher": None,
}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _as_list(value: Any) -> List[str]:
    """Normalise a CLI/env value into a list of stripped strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value]
    else:
        items = str(value).split(",")
    return [item.strip() for item in items if str(item).strip()]


def _as_floats(value: Any, default: Sequence[float]) -> List[float]:
    items = _as_list(value)
    if not items:
        return [float(v) for v in default]
    return [float(v) for v in items]


def _as_ints(value: Any, default: Sequence[int]) -> List[int]:
    items = _as_list(value)
    if not items:
        return [int(v) for v in default]
    return [int(float(v)) for v in items]


def _unwrap(cfg: Dict[str, Any], *names: str) -> Dict[str, Any]:
    """Fetch the first nested config section present (supports flat and nested dicts)."""
    for name in names:
        section = cfg.get(name)
        if isinstance(section, dict):
            return section
    return {}


def _obs_action_dims(env: Any) -> Tuple[int, int, bool]:
    """Best-effort (obs_dim, action_dim, discrete) extraction from a gym env."""
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    obs_dim = 1
    action_dim = 1
    discrete = False
    shape = getattr(obs_space, "shape", None)
    if shape is not None:
        obs_dim = int(np.prod(shape)) if len(tuple(shape)) else 1
    elif hasattr(obs_space, "n"):
        obs_dim = int(obs_space.n)
    if hasattr(act_space, "n"):
        discrete = True
        action_dim = int(act_space.n)
    else:
        shape = getattr(act_space, "shape", None)
        if shape is not None:
            action_dim = int(np.prod(shape)) if len(tuple(shape)) else 1
    return int(obs_dim), int(action_dim), bool(discrete)


def _safe(fn: Callable[..., Any], *args: Any, default: Any = None, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def mask_budget_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Fixed mask-training sample budget for the efficiency comparison (Table 4)."""
    key = normalize_env_key(env_id) if _HAS_POLICIES else env_id
    if _HAS_STATEMASK:
        budget = _safe(samples_for, key, default=None)
        if budget:
            return int(budget)
    if key in TABLE4_SAMPLES:
        return int(TABLE4_SAMPLES[key])
    section = _unwrap(cfg or {}, "explanation", "mask", "mask_trainer")
    return int(section.get("total_timesteps", 300_000))


def d_max_of(env: Any, env_id: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Max single-episode reward used to normalise the fidelity score."""
    section = _unwrap(cfg or {}, "env")
    if section.get("d_max") is not None:
        return float(section["d_max"])
    if _HAS_ENVS:
        value = _safe(d_max_for, env_id, default=None)
        if value:
            return float(value)
    return None


# --------------------------------------------------------------------------------------
# Environment / policy construction
# --------------------------------------------------------------------------------------
def build_experiment_env(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    mode: str = "eval",
) -> Any:
    """Create the environment used by Experiment I (frozen-policy rollouts)."""
    if not _HAS_ENVS:
        raise RuntimeError("rice.envs.make_env is unavailable; cannot build environments.")
    cfg = cfg or {}
    env_section = _unwrap(cfg, "env")
    normalize = env_section.get("normalize_obs")
    max_steps = env_section.get("max_episode_steps")
    env = make_env(
        env_id,
        seed=seed,
        normalize=normalize,
        mode=mode,
        max_episode_steps=max_steps,
    )
    return env


def build_or_load_policy(
    env: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Any:
    """Load the frozen pre-trained policy ``pi`` (or build a fresh one if no checkpoint)."""
    if not _HAS_POLICIES:
        raise RuntimeError("rice.models.policies is unavailable; cannot build the target policy.")
    cfg = cfg or {}
    obs_dim, action_dim, discrete = _obs_action_dims(env)
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)

    if checkpoint is None:
        checkpoint = _unwrap(cfg, "target", "policy").get("checkpoint")
    if checkpoint and os.path.exists(checkpoint):
        policy = _safe(
            load_policy,
            checkpoint,
            env_id=env_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            default=None,
        )
        if policy is not None:
            if logger:
                logger.info("loaded pre-trained target policy from %s", checkpoint)
            return policy
    if logger:
        logger.warning(
            "no pre-trained checkpoint for %s (looked for %s); using a freshly initialised "
            "policy -- reproduce trends after running scripts/train_target.py",
            env_id,
            checkpoint,
        )
    return build_policy(
        env_id=env_id,
        obs_dim=obs_dim,
        action_dim=action_dim,
        observation_space=obs_space,
        action_space=act_space,
        kind="policy",
        discrete=discrete,
        device=device,
    )


def pretrain_target_policy(
    env: Any,
    env_id: str,
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    device: str = "cpu",
    logger: Optional[Any] = None,
) -> Any:
    """Pre-train the (sub-optimal) target policy with plain PPO.

    Reuses the Stage-2 PPO engine with *both* RICE contributions disabled
    (``use_mixed_init=False`` and ``use_rnd=False``), which is exactly standard PPO
    pre-training.  This keeps the "No Refine" regime of Table 1 reproducible without
    duplicating a PPO implementation here.
    """
    try:
        from rice.refining.ppo_refine import refine_policy
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"cannot pre-train the target policy: {exc}") from exc
    policy, _trainer = refine_policy(
        env,
        policy=None,
        total_timesteps=total_timesteps,
        env_id=env_id,
        config={
            "use_mixed_init": False,
            "use_rnd": False,
            "p": 0.0,
            "lam": 0.0,
            "seed": seed,
        },
        seed=seed,
        device=device,
        logger=logger,
        progress=False,
    )
    return policy


# --------------------------------------------------------------------------------------
# Stage-1: train the explanation methods and time them (Table 4)
# --------------------------------------------------------------------------------------
@dataclass
class ExplanationArtifacts:
    """Trained explanation for one method + its wall-clock training cost."""

    method: str
    env_id: str
    mask_net: Optional[Any] = None
    trainer: Optional[Any] = None
    train_time: Optional[float] = None
    samples: Optional[int] = None
    seconds_per_sample: Optional[float] = None
    checkpoint: Optional[str] = None
    scoring: str = "auto"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "env_id": self.env_id,
            "train_time": self.train_time,
            "samples": self.samples,
            "seconds_per_sample": self.seconds_per_sample,
            "checkpoint": self.checkpoint,
            "scoring": self.scoring,
            **{k: v for k, v in self.extra.items() if isinstance(v, (int, float, str, bool, type(None)))},
        }


def train_ours_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: int,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> ExplanationArtifacts:
    """Stage 1 for RICE: Algorithm 1 (vanilla PPO + ``alpha * a_t^m`` blinding bonus)."""
    if not _HAS_MASK_TRAINER:
        raise RuntimeError("rice.explanation.mask_trainer is unavailable.")
    explanation_cfg = _unwrap(cfg or {}, "explanation", "mask", "mask_trainer")
    mask_net, trainer = train_mask_network(
        env,
        policy,
        total_timesteps=total_timesteps,
        alpha=alpha,
        env_id=env_id,
        config=explanation_cfg or None,
        logger=logger,
        save_path=checkpoint,
        seed=seed,
        device=device,
        store_dataset=False,
        progress=False,
    )
    total_time = float(getattr(trainer, "total_time", 0.0) or 0.0)
    per_sample = _safe(lambda: float(trainer.seconds_per_sample), default=None)
    return ExplanationArtifacts(
        method="ours",
        env_id=env_id,
        mask_net=mask_net,
        trainer=trainer,
        train_time=total_time,
        samples=int(total_timesteps),
        seconds_per_sample=per_sample,
        checkpoint=checkpoint,
        scoring="mask",
        extra={"alpha": alpha},
    )


def train_statemask_explanation(
    env: Any,
    policy: Any,
    env_id: str,
    total_timesteps: int,
    alpha: float = 0.01,
    seed: int = 0,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    logger: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> ExplanationArtifacts:
    """StateMask mask training (primal-dual ``min |eta(pi) - eta(pi_bar)|``)."""
    if not _HAS_STATEMASK:
        raise RuntimeError("rice.baselines.statemask_r is unavailable.")
    sm_cfg = _unwrap(cfg or {}, "statemask", "state_mask")
    mask_net, trainer = train_statemask_network(
        env,
        policy,
        total_timesteps=total_timesteps,
        alpha=alpha,
        env_id=env_id,
        config=sm_cfg or None,
        logger=logger,
        save_path=checkpoint,
        seed=seed,
        device=device,
        store_dataset=False,
        progress=False,
    )
    total_time = float(getattr(trainer, "total_time", 0.0) or 0.0)
    per_sample = _safe(lambda: float(trainer.seconds_per_sample), default=None)
    return ExplanationArtifacts(
        method="statemask",
        env_id=env_id,
        mask_net=mask_net,
        trainer=trainer,
        train_time=total_time,
        samples=int(total_timesteps),
        seconds_per_sample=per_sample,
        checkpoint=checkpoint,
        scoring="mask",
        extra={"alpha": alpha},
    )


def random_explanation(env_id: str, logger: Optional[Any] = None) -> ExplanationArtifacts:
    """Random explanation baseline: no mask network, uninformative importance scores."""
    if logger:
        logger.info("Random explanation needs no training (uniform importance).")
    return ExplanationArtifacts(method="random", env_id=env_id, train_time=0.0, samples=0, scoring="random")


def train_explanation(
    method: str,
    env: Any,
    policy: Any,
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    device: str = "cpu",
    logger: Optional[Any] = None,
    mask_timesteps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
) -> ExplanationArtifacts:
    """Dispatch mask training for one explanation method with a fixed sample budget."""
    explanation_cfg = _unwrap(cfg or {}, "explanation", "mask", "mask_trainer")
    budget = int(mask_timesteps or mask_budget_for(env_id, cfg))
    alpha = float(explanation_cfg.get("alpha", DEFAULT_ALPHA))
    method = method.strip().lower()

    checkpoint = None
    if checkpoint_dir:
        checkpoint = os.path.join(ensure_dir(checkpoint_dir), f"{env_id}_{method}_mask.pt")

    if method in ("ours", "rice", "mask", "vanilla", "vanilla_ppo"):
        return train_ours_explanation(
            env, policy, env_id, budget,
            alpha=alpha, seed=seed, device=device,
            checkpoint=checkpoint, logger=logger, cfg=cfg,
        )
    if method in ("statemask", "state_mask", "sm"):
        return train_statemask_explanation(
            env, policy, env_id, budget,
            alpha=float(_unwrap(cfg or {}, "statemask").get("alpha_init", 0.01)),
            seed=seed, device=device,
            checkpoint=checkpoint, logger=logger, cfg=cfg,
        )
    if method in ("random", "rand"):
        return random_explanation(env_id, logger=logger)
    raise ValueError(f"unknown explanation method: {method!r}")


# --------------------------------------------------------------------------------------
# Fidelity evaluation (paper Eq. of Sec. 4.1 / StateMask metric)
# --------------------------------------------------------------------------------------
def evaluate_fidelity_for_method(
    env: Any,
    policy: Any,
    artifacts: ExplanationArtifacts,
    env_id: str,
    K_values: Sequence[float] = DEFAULT_K_VALUES,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    d_max: Optional[float] = None,
    deterministic_policy: bool = True,
    store_details: bool = False,
    progress: bool = False,
    logger: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[float, Any]:
    """Evaluate the fidelity of one explanation over ``K in K_values`` (500 traj x 3 seeds)."""
    if not _HAS_FIDELITY:
        raise RuntimeError("rice.explanation.fidelity is unavailable.")
    fidelity_cfg = _unwrap(cfg or {}, "fidelity")
    scoring = artifacts.scoring
    if scoring == "auto":
        scoring = "mask" if artifacts.mask_net is not None else "random"

    results: Dict[float, Any] = {}
    for K in K_values:
        results[float(K)] = evaluate_fidelity(
            env,
            policy,
            mask_net=artifacts.mask_net,
            K=float(K),
            n_trajectories=int(fidelity_cfg.get("n_trajectories", n_trajectories)),
            seeds=tuple(fidelity_cfg.get("seeds", seeds)),
            env_id=env_id,
            d_max=d_max,
            scoring=scoring,
            deterministic_policy=bool(fidelity_cfg.get("deterministic_policy", deterministic_policy)),
            progress=progress,
            store_details=bool(fidelity_cfg.get("store_details", store_details)),
            logger=logger,
            seed=int(seeds[0]) if len(seeds) else 0,
        )
    return results


def summarize_fidelity(results: Dict[float, Any]) -> Dict[str, str]:
    """Render one method's per-K results as ``{"10%": "mean +- std", ...}``."""
    out: Dict[str, str] = {}
    for K, res in sorted(results.items(), key=lambda kv: kv[0]):
        key = f"{int(round(float(K) * 100))}%"
        try:
            out[key] = res.format(decimals=3, percent=True)
        except Exception:
            mean = float(getattr(res, "mean", float("nan")))
            std = float(getattr(res, "std", float("nan")))
            out[key] = format_mean_std([mean, mean - std if std == std else mean], decimals=3)
    return out


# --------------------------------------------------------------------------------------
# Efficiency comparison (Table 4)
# --------------------------------------------------------------------------------------
def efficiency_report(
    env_id: str,
    artifacts: Dict[str, ExplanationArtifacts],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare mask-network training time of ``ours`` vs ``statemask`` (Sec. 4.3)."""
    ours = artifacts.get("ours")
    sm = artifacts.get("statemask")
    report: Dict[str, Any] = {
        "env_id": env_id,
        "samples": (ours.samples if ours else (sm.samples if sm else None)),
        "ours_seconds": ours.train_time if ours else None,
        "statemask_seconds": sm.train_time if sm else None,
        "paper_expected_reduction": PAPER_TIME_REDUCTION,
        "reference": TABLE4_TIMES.get(normalize_env_key(env_id) if _HAS_POLICIES else env_id),
    }
    if ours and sm and ours.train_time and sm.train_time and sm.train_time > 0:
        report["reduction"] = float(1.0 - float(ours.train_time) / float(sm.train_time))
        if _HAS_FIDELITY:
            extra = _safe(
                training_time_reduction,
                {"statemask": float(sm.train_time)},
                {"ours": float(ours.train_time)},
                default=None,
            )
            if isinstance(extra, dict):
                report["training_time_reduction"] = extra
    else:
        report["reduction"] = None
    return report


# --------------------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------------------
def run_experiment1(
    env_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    methods: Sequence[str] = EXPLANATION_METHODS,
    K_values: Sequence[float] = DEFAULT_K_VALUES,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    mask_timesteps: Optional[int] = None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Optional[Any] = None,
    progress: bool = False,
    pretrain_timesteps: Optional[int] = None,
    checkpoint: Optional[str] = None,
    store_details: bool = False,
) -> Dict[str, Any]:
    """Run Experiment I for a single application."""
    cfg = cfg or {}
    logger = logger or get_logger("rice.exp1")
    key = normalize_env_key(env_id) if _HAS_POLICIES else env_id
    logger.info("=== Experiment I: fidelity & efficiency on %s ===", key)

    seeds = list(seeds)
    set_seed(int(seeds[0]) if seeds else 0)

    env = build_experiment_env(env_id, cfg=cfg, seed=int(seeds[0]) if seeds else 0, mode="eval")
    backend = _safe(env_backend, env, default="unknown") if _HAS_ENVS else "unknown"
    meta = _safe(env_metadata, env_id, default={}) if _HAS_ENVS else {}
    d_max = d_max_of(env, env_id, cfg=cfg)

    policy = build_or_load_policy(env, env_id, cfg=cfg, device=device, checkpoint=checkpoint, logger=logger)
    if pretrain_timesteps:
        logger.info("pre-training target policy for %s steps (No-Refine regime)", pretrain_timesteps)
        policy = pretrain_target_policy(
            env, env_id, total_timesteps=int(pretrain_timesteps), seed=int(seeds[0]) if seeds else 0,
            device=device, logger=logger,
        )

    checkpoint_dir = os.path.join(out_dir, "checkpoints") if out_dir else None
    artifacts: Dict[str, ExplanationArtifacts] = {}
    for method in methods:
        method_key = method.strip().lower()
        logger.info("training explanation %r (budget=%s samples)", method_key, mask_timesteps or mask_budget_for(env_id, cfg))
        t0 = time.time()
        try:
            artifacts[method_key] = train_explanation(
                method_key, env, policy, env_id, cfg=cfg,
                seed=int(seeds[0]) if seeds else 0,
                device=device, logger=logger,
                mask_timesteps=mask_timesteps, checkpoint_dir=checkpoint_dir,
            )
        except Exception as exc:  # keep the driver usable if a baseline is unavailable
            logger.warning("explanation %r failed (%s); skipping", method_key, exc)
            continue
        logger.info("  -> %s training time: %.1fs", method_key, time.time() - t0)

    fidelity: Dict[str, Dict[str, Any]] = {}
    summaries: Dict[str, Dict[str, str]] = {}
    for method_key, art in artifacts.items():
        logger.info("evaluating fidelity for %r over K=%s", method_key, list(K_values))
        try:
            results = evaluate_fidelity_for_method(
                env, policy, art, env_id,
                K_values=K_values, n_trajectories=n_trajectories, seeds=seeds,
                d_max=d_max, progress=progress, logger=logger, cfg=cfg,
                store_details=store_details,
            )
        except Exception as exc:
            logger.warning("fidelity evaluation for %r failed (%s)", method_key, exc)
            continue
        fidelity[method_key] = {
            (f"{int(round(float(K) * 100))}%"): res.to_dict(include_details=store_details)
            for K, res in sorted(results.items(), key=lambda kv: kv[0])
        }
        summaries[method_key] = summarize_fidelity(results)

    efficiency = efficiency_report(env_id, artifacts, cfg=cfg)

    report: Dict[str, Any] = {
        "experiment": "exp1_fidelity_efficiency",
        "env_id": key,
        "env_backend": backend,
        "env_metadata": meta if isinstance(meta, dict) else {},
        "d_max": d_max,
        "K_values": [float(k) for k in K_values],
        "n_trajectories": int(n_trajectories),
        "seeds": [int(s) for s in seeds],
        "mask_timesteps": {m: a.samples for m, a in artifacts.items()},
        "fidelity": fidelity,
        "fidelity_summary": summaries,
        "efficiency": efficiency,
        "training_times": {m: a.train_time for m, a in artifacts.items()},
        "reference_no_refine_return": REFERENCE_NO_REFINE.get(key),
    }

    if out_dir:
        ensure_dir(out_dir)
        save_json(report, os.path.join(out_dir, f"exp1_{key}.json"))
        logger.info("wrote %s", os.path.join(out_dir, f"exp1_{key}.json"))
    _close_env(env)
    return report


def _close_env(env: Any) -> None:
    _safe(getattr(env, "close", lambda: None))


def run_experiment1_multi(
    env_ids: Sequence[str],
    cfg: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Experiment I across several applications (paper reports all of them)."""
    out: Dict[str, Any] = {}
    for env_id in env_ids:
        try:
            out[env_id] = run_experiment1(env_id, cfg=cfg, **kwargs)
        except Exception as exc:  # pragma: no cover - keep the sweep alive
            logger = kwargs.get("logger") or get_logger("rice.exp1")
            logger.warning("Experiment I failed for %s: %s", env_id, exc)
            out[env_id] = {"env_id": env_id, "error": str(exc)}
    return out


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def format_report(report: Dict[str, Any]) -> str:
    """Human-readable Table-1/Table-4-style rendering of one Experiment-I report."""
    lines: List[str] = []
    lines.append(f"Experiment I -- {report.get('env_id')} (d_max={report.get('d_max')})")
    lines.append(f"  trajectories: {report.get('n_trajectories')}  seeds: {report.get('seeds')}")
    lines.append("  Fidelity (mean +- std, higher is better):")
    summary = report.get("fidelity_summary", {})
    if summary:
        header = "    {:<12}".format("method") + "".join(
            "{:>22}".format(k) for k in sorted(summary[next(iter(summary))].keys(), key=lambda s: int(s.rstrip('%')))
        )
        lines.append(header)
        for method, per_k in summary.items():
            row = "    {:<12}".format(method)
            for k in sorted(per_k.keys(), key=lambda s: int(s.rstrip('%'))):
                row += "{:>22}".format(str(per_k[k]))
            lines.append(row)
    else:
        lines.append("    (no fidelity results)")
    eff = report.get("efficiency", {})
    lines.append("  Mask-network training time (Table 4):")
    lines.append(f"    samples: {eff.get('samples')}")
    lines.append(f"    ours:      {eff.get('ours_seconds')} s")
    lines.append(f"    statemask: {eff.get('statemask_seconds')} s")
    red = eff.get("reduction")
    if red is not None:
        lines.append(
            "    reduction: {:.1%} (paper reports ~{:.1%})".format(float(red), PAPER_TIME_REDUCTION)
        )
    else:
        lines.append("    reduction: n/a (train both methods to compare)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RICE Experiment I -- fidelity & efficiency of the explanation method",
    )
    parser.add_argument("--config", default=None, help="config stem to load (e.g. hopper, cage2)")
    parser.add_argument("--env", "--envs", dest="envs", default=None,
                        help="comma separated env ids (default: all applications)")
    parser.add_argument("--methods", default="ours,statemask,random",
                        help="comma separated explanation methods")
    parser.add_argument("--K", "--K-values", dest="K_values", default=None,
                        help="comma separated fidelity window fractions (default 0.1,0.2,0.3,0.4)")
    parser.add_argument("--n-trajectories", type=int, default=None,
                        help="trajectories per (method, K, seed) -- paper uses 500")
    parser.add_argument("--seeds", default=None, help="comma separated seed list (paper uses 3 seeds)")
    parser.add_argument("--mask-timesteps", type=int, default=None,
                        help="fixed mask-training sample budget (defaults to Table 4)")
    parser.add_argument("--pretrain-timesteps", type=int, default=None,
                        help="optionally pre-train the target policy for N steps first")
    parser.add_argument("--checkpoint", default=None, help="path to a pre-trained target policy")
    parser.add_argument("--out", "--out-dir", dest="out_dir", default=None, help="results directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 10 trajectories, 1 seed, K=10%%, 2000 mask samples")
    parser.add_argument("--store-details", action="store_true", help="keep per-trajectory details")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the raw report as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config:
        try:
            cfg = get_config(args.config)
        except Exception as exc:
            print(f"[warn] could not load config {args.config!r}: {exc}", file=sys.stderr)

    envs = _as_list(args.envs) or [e for e in _as_list(_unwrap(cfg, "experiments").get("envs"))] or list(DEFAULT_ENVS)
    if args.quick and not args.envs:
        envs = [envs[0]]

    methods = _as_list(args.methods) or list(EXPLANATION_METHODS)
    K_values = _as_floats(args.K_values, DEFAULT_K_VALUES)
    n_trajectories = args.n_trajectories or (10 if args.quick else DEFAULT_N_TRAJECTORIES)
    seeds = _as_ints(args.seeds, (0,) if args.quick else DEFAULT_SEEDS)
    mask_timesteps = args.mask_timesteps or (2_000 if args.quick else None)
    out_dir = args.out_dir or os.path.join("results", "exp1_fidelity_efficiency")

    logger = get_logger("rice.exp1", out_dir=out_dir)
    logger.info(
        "Experiment I | envs=%s methods=%s K=%s n_traj=%s seeds=%s mask_samples=%s",
        envs, methods, K_values, n_trajectories, seeds, mask_timesteps,
    )

    reports = run_experiment1_multi(
        envs,
        cfg=cfg,
        methods=methods,
        K_values=K_values,
        n_trajectories=n_trajectories,
        seeds=seeds,
        mask_timesteps=mask_timesteps,
        device=args.device,
        out_dir=out_dir,
        logger=logger,
        progress=args.progress,
        pretrain_timesteps=args.pretrain_timesteps,
        checkpoint=args.checkpoint,
        store_details=args.store_details,
    )

    for env_id, report in reports.items():
        if "error" in report:
            logger.warning("%s: %s", env_id, report["error"])
            continue
        print(format_report(report))
    if args.json:
        print(json.dumps(reports, indent=2, default=str))

    if out_dir:
        save_json(reports, os.path.join(out_dir, "exp1_all.json"))
        logger.info("summary written to %s", os.path.join(out_dir, "exp1_all.json"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
