"""FRE (Functional Reward Encodings) -- top-level entry point.

This script orchestrates the whole FRE reproduction pipeline described in the
paper (ICML 2024, "Zero-Shot Reinforcement Learning via Functional Reward
Encodings"):

    Phase 1 (Algorithm 1, "# Train encoder"):
        Sample eta ~ p(eta), sample K encoder states and K' decoder states from
        the unlabeled offline dataset D, and train the permutation-invariant
        transformer encoder + MLP decoder by maximizing Eq. (6)
        (reconstruction log-likelihood minus beta * KL to the unit Gaussian).
        The RL components are *not* trained during this phase.

    Phase 2 (Algorithm 1, "# Train policy"):
        Freeze the encoder.  Sample eta ~ p(eta), encode
        z ~ p_theta(z | {(s^e_k, eta(s^e_k))}) from K = 32 context pairs and
        train pi(a|s,z), Q(s,a,z), V(s,z) with IQL using r = eta(s).  Freezing
        keeps the eta -> z mapping stationary for TD learning.

    Zero-shot evaluation (Sec. 5 / Table 1):
        Encode a *new* downstream task from only 32 (state, reward) samples,
        then roll out the conditioned policy online for 20 episodes.  Results
        are normalized to 0-100 and aggregated as mean +- std over 5 seeds.
        (FB / SF need 5120 reward samples; OPAL-10 is evaluated privileged over
        10 skills with online rollouts.)

Usage examples
--------------
    # full FRE run (train + evaluate) on AntMaze
    python fre/main.py --domain antmaze --agent fre --stage all

    # only train the encoder / only train the policy
    python fre/main.py --domain antmaze --stage encoder
    python fre/main.py --domain antmaze --stage policy

    # evaluate a saved checkpoint (5 seeds x 20 episodes)
    python fre/main.py --domain exorl_walker --stage eval --checkpoint runs/fre/...

    # prior-mixture ablation (Table 4)
    python fre/main.py --domain antmaze --prior-mixture goals
    python fre/main.py --domain antmaze --prior-mixture goal-lin

    # baselines
    python fre/main.py --domain antmaze --agent gc_iql
    python fre/main.py --domain antmaze --agent gc_bc
    python fre/main.py --domain antmaze --agent opal
    python fre/main.py --domain antmaze --agent fb      # requires controllable_agent

    # quick smoke test without D4RL/GPU
    python fre/main.py --domain antmaze --allow-synthetic --quick --num-eval-episodes 1

Everything heavy (torch / gym / d4rl) is imported lazily so that ``--help`` and
``--dry-run`` work in a bare environment.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Make the repository root importable when this file is executed directly
# (``python fre/main.py ...``).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fre.config.envs import (  # noqa: E402  (import after sys.path fix)
    ANTMAZE,
    DOMAINS,
    KITCHEN,
    TABLE1_REFERENCE,
    default_domain_overrides,
    get_domain_config,
    get_eval_tasks,
    make_config,
)
from fre.utils.logging import (  # noqa: E402
    MetricLogger,
    TableLogger,
    get_logger,
    make_run_dir,
    save_json,
)
from fre.utils.normalization import (  # noqa: E402
    TABLE1_AGGREGATE_TARGETS,
    TABLE1_FRE_TARGETS,
    TABLE4_FRE_TARGETS,
    ResultAccumulator,
    format_mean_std,
)

LOG = get_logger("fre.main")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENTS = ("fre", "gc_iql", "gc_bc", "opal", "fb", "sf")
STAGES = ("encoder", "policy", "train", "eval", "all", "dry-run")

#: Domains grouped under a friendly alias (``--domain exorl`` == both ExORL
#: domains, ``--domain all`` == every domain in the paper).
DOMAIN_GROUPS: Dict[str, List[str]] = {
    "antmaze": ["antmaze"],
    "ant": ["antmaze"],
    "exorl": ["exorl_walker", "exorl_cheetah"],
    "kitchen": ["kitchen"],
    "all": ["antmaze", "exorl_walker", "exorl_cheetah", "kitchen"],
}

#: Paper Table 4 prior-mixture ablation names -> mixture weights.
PRIOR_MIXTURES: Dict[str, Dict[str, float]] = {
    "all": {"goal": 1.0, "linear": 1.0, "mlp": 1.0},
    "goals": {"goal": 1.0, "linear": 0.0, "mlp": 0.0},
    "goal": {"goal": 1.0, "linear": 0.0, "mlp": 0.0},
    "lin": {"goal": 0.0, "linear": 1.0, "mlp": 0.0},
    "linear": {"goal": 0.0, "linear": 1.0, "mlp": 0.0},
    "mlp": {"goal": 0.0, "linear": 0.0, "mlp": 1.0},
    "lin-mlp": {"goal": 0.0, "linear": 0.5, "mlp": 0.5},
    "goal-mlp": {"goal": 0.5, "linear": 0.0, "mlp": 0.5},
    "goal-lin": {"goal": 0.5, "linear": 0.5, "mlp": 0.0},
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def call_with_supported_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` after dropping keyword arguments it does not accept.

    The five evaluation suites, the trainers and the baselines all expose
    slightly different signatures.  Rather than hard-coding every variant, we
    filter the kwargs against the callee's signature (functions accepting
    ``**kwargs`` receive everything).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C functions
        return fn(*args, **kwargs)

    params = sig.parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(*args, **kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


def set_global_seed(seed: int) -> None:
    """Seed numpy / torch / stdlib RNGs for a reproducible evaluation seed."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is a hard dep in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional for --help
        pass


def resolve_domains(name: str) -> List[str]:
    """Expand a domain name (or alias) into canonical domain names."""
    key = str(name).strip().lower()
    if key in DOMAIN_GROUPS:
        return list(DOMAIN_GROUPS[key])
    if key in DOMAINS:
        cfg = DOMAINS[key]
        return [cfg.name]
    try:
        return [get_domain_config(key).name]
    except Exception:
        pass
    raise ValueError(
        f"Unknown domain {name!r}. Known domains: {sorted(set(DOMAINS))} "
        f"and groups {sorted(DOMAIN_GROUPS)}"
    )


def lookup_reference(agent: str, row: str) -> Optional[Tuple[float, float]]:
    """Best-effort lookup of a published Table 1 number for ``row``.

    ``fre.config.envs.TABLE1_REFERENCE`` is a nested structure; depending on how
    it was built the agent name may sit at the outer or inner level, so probe
    several plausible layouts.
    """
    tables: List[Any] = [TABLE1_REFERENCE, TABLE1_FRE_TARGETS, TABLE1_AGGREGATE_TARGETS]
    for table in tables:
        if not isinstance(table, dict):
            continue
        # {agent: {row: (mean, std)}}
        sub = table.get(agent)
        if isinstance(sub, dict) and row in sub:
            return _as_pair(sub[row])
        # {row: {agent: (mean, std)}}
        sub = table.get(row)
        if isinstance(sub, dict):
            for key in (agent, agent.upper(), agent.replace("_", "-")):
                if key in sub:
                    return _as_pair(sub[key])
        # {row: (mean, std)}  (FRE-specific tables)
        if row in table and agent == "fre":
            pair = _as_pair(table[row])
            if pair is not None:
                return pair
    return None


def _as_pair(value: Any) -> Optional[Tuple[float, float]]:
    """Coerce a table cell into a ``(mean, std)`` tuple when possible."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        mean = value.get("mean", value.get("score"))
        std = value.get("std", value.get("score_std", 0.0))
        if mean is not None:
            try:
                return float(mean), float(std or 0.0)
            except (TypeError, ValueError):
                return None
    try:
        return float(value), 0.0
    except (TypeError, ValueError):
        return None


def _flatten_tasks(obj: Any) -> List[Tuple[str, Any]]:
    """Flatten a suite dict/list into an ordered ``[(name, task), ...]`` list."""
    out: List[Tuple[str, Any]] = []
    if obj is None:
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    name = getattr(item, "name", None) or (
                        key if len(value) == 1 else f"{key}-{i}"
                    )
                    out.append((str(name), item))
            else:
                name = getattr(value, "name", None) or str(key)
                out.append((str(name), value))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            out.append((str(getattr(item, "name", None) or f"task-{i}"), item))
    return out


def _dataset_obs(dataset: Any) -> Any:
    """Return the state array of a ReplayBuffer-like object."""
    for attr in ("states", "observations", "obs"):
        arr = getattr(dataset, attr, None)
        if arr is not None:
            return arr
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fre",
        description=(
            "Zero-Shot Reinforcement Learning via Functional Reward Encodings -- "
            "train the FRE encoder/decoder, train the FRE-conditioned IQL policy, "
            "and/or evaluate zero-shot on AntMaze / ExORL / Kitchen."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- what to run -------------------------------------------------------
    p.add_argument(
        "--domain",
        default="antmaze",
        help="antmaze | exorl_walker | exorl_cheetah | kitchen | exorl | all",
    )
    p.add_argument(
        "--agent",
        default="fre",
        choices=AGENTS,
        help="Which method to train/evaluate (fre is the paper's contribution).",
    )
    p.add_argument(
        "--stage",
        default="all",
        choices=STAGES,
        help=(
            "encoder = phase 1 only; policy = phase 2 only; train = both phases; "
            "eval = evaluate only; all = train + evaluate; dry-run = print plan."
        ),
    )

    # --- data --------------------------------------------------------------
    p.add_argument("--dataset-path", default=None, help="Explicit offline dataset file.")
    p.add_argument("--dataset-dir", default=None, help="Directory searched for datasets.")
    p.add_argument(
        "--exorl-variant",
        default="rnd",
        help="ExORL dataset variant (paper uses the RND datasets).",
    )
    p.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Fall back to a synthetic random dataset when D4RL/ExORL is unavailable.",
    )
    p.add_argument(
        "--no-physics",
        action="store_true",
        help="Disable the ExORL physics augmentation of the encoder input (Appendix C.2).",
    )
    p.add_argument(
        "--synthetic-transitions",
        type=int,
        default=20_000,
        help="Size of the synthetic fallback dataset.",
    )

    # --- training schedule (Table 3) ---------------------------------------
    p.add_argument("--encoder-steps", type=int, default=None,
                   help="Override encoder steps (Table 3: 150k; 1M ExORL/Kitchen).")
    p.add_argument("--policy-steps", type=int, default=None,
                   help="Override policy steps (Table 3: 850k; 1M ExORL/Kitchen).")
    p.add_argument("--scale-steps", type=float, default=None,
                   help="Scale both step budgets (e.g. 0.001 for a smoke test).")
    p.add_argument("--batch-size", type=int, default=None, help="Table 3: 512.")
    p.add_argument("--learning-rate", type=float, default=None, help="Table 3: 1e-4.")
    p.add_argument("--beta-kl", type=float, default=None, help="Table 3: 0.01.")
    p.add_argument("--discount", type=float, default=None, help="Table 3: 0.88.")
    p.add_argument("--expectile", type=float, default=None, help="Table 3: 0.8.")
    p.add_argument("--temperature", type=float, default=None, help="Table 3: 3.0.")
    p.add_argument("--target-update-rate", type=float, default=None, help="Table 3: 0.001.")
    p.add_argument("--num-encoder-samples", type=int, default=None, help="Table 3: K=32.")
    p.add_argument("--num-decoder-samples", type=int, default=None, help="Table 3: K'=8.")
    p.add_argument(
        "--prior-mixture",
        default="all",
        choices=sorted(PRIOR_MIXTURES),
        help="Prior reward distribution mixture (Table 4 ablation).",
    )
    p.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="Optional global scale applied to sampled prior rewards.",
    )

    # --- evaluation --------------------------------------------------------
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds to run (default: 0..num-eval-seeds-1; paper uses 5).")
    p.add_argument("--num-eval-seeds", type=int, default=None,
                   help="Number of seeds when --seeds is not given (paper: 5).")
    p.add_argument("--num-eval-episodes", type=int, default=None,
                   help="Episodes per task per seed (paper: 20).")
    p.add_argument("--suites", nargs="+", default=None,
                   help="Evaluation suites to run (default: the whole domain).")
    p.add_argument("--eval-samples", type=int, default=None,
                   help="Reward samples used to encode z (FRE/GC: 32; FB/SF: 5120).")
    p.add_argument("--no-eval", action="store_true", help="Skip online evaluation.")
    p.add_argument("--deterministic-eval", dest="deterministic_eval", action="store_true",
                   default=True, help="Use the policy mean at evaluation time.")
    p.add_argument("--stochastic-eval", dest="deterministic_eval", action="store_false",
                   help="Sample from the policy at evaluation time.")

    # --- bookkeeping -------------------------------------------------------
    p.add_argument("--output-dir", default=None, help="Run directory (default: runs/fre).")
    p.add_argument("--tag", default=None, help="Optional run-name tag.")
    p.add_argument("--checkpoint", default=None, help="Checkpoint to resume/evaluate from.")
    p.add_argument("--save-checkpoint", default=None,
                   help="Where to write the trained checkpoint (default: <run-dir>/ckpt.pt).")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    p.add_argument("--seed", type=int, default=0, help="Base seed for training.")
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke-test run (few steps, 1 seed, 1 episode).")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit.")
    p.add_argument("--json", default=None, help="Write the final report to this JSON file.")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)

    if args.quick:
        args.allow_synthetic = True
        if args.encoder_steps is None:
            args.encoder_steps = 200
        if args.policy_steps is None:
            args.policy_steps = 200
        if args.synthetic_transitions is None or args.synthetic_transitions == 20_000:
            args.synthetic_transitions = 2_000
        if args.seeds is None and args.num_eval_seeds is None:
            args.num_eval_seeds = 1
        if args.num_eval_episodes is None:
            args.num_eval_episodes = 1
        if args.eval_interval == 10_000:
            args.eval_interval = 10_000  # keep, but the run is short anyway
    return args


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def config_overrides_from_args(args: argparse.Namespace, domain: str) -> Dict[str, Any]:
    """Translate CLI flags into ``fre.config.default.Config`` overrides."""
    domain_overrides = default_domain_overrides(domain)
    overrides: Dict[str, Any] = dict(domain_overrides)

    overrides["seed"] = int(args.seed)
    if args.device:
        overrides["device"] = args.device

    # Table 3 hyperparameters (only when explicitly overridden).
    for attr in (
        "batch_size",
        "learning_rate",
        "beta_kl",
        "discount",
        "expectile",
        "temperature",
        "target_update_rate",
        "num_encoder_samples",
        "num_decoder_samples",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            overrides[attr] = value
    if args.expectile is not None:
        overrides["iql_expectile"] = args.expectile
    if args.temperature is not None:
        overrides["iql_temperature"] = args.temperature
    if args.beta_kl is not None:
        overrides["beta_kl"] = args.beta_kl

    # Prior mixture (Table 4 ablation).
    mixture = PRIOR_MIXTURES.get(str(args.prior_mixture).lower())
    if mixture is not None:
        total = sum(mixture.values())
        if total > 0:
            overrides["prior_ratios"] = {k: v / total for k, v in mixture.items()}
    if args.reward_scale is not None:
        overrides["prior_reward_scale"] = args.reward_scale

    # Evaluation protocol.
    if args.num_eval_episodes is not None:
        overrides["num_eval_episodes"] = int(args.num_eval_episodes)
    num_seeds = args.num_eval_seeds
    if num_seeds is not None:
        overrides["num_eval_seeds"] = int(num_seeds)

    # Dataset plumbing.
    if args.dataset_path:
        overrides["dataset_path"] = args.dataset_path
    if args.dataset_dir:
        overrides["dataset_dir"] = args.dataset_dir
    if args.exorl_variant:
        overrides["exorl_variant"] = args.exorl_variant
    if args.no_physics:
        overrides["use_physics_augmentation"] = False
    if args.eval_samples is not None:
        overrides["fre_eval_samples"] = int(args.eval_samples)
    return overrides


def build_config(domain: str, args: argparse.Namespace, **extra: Any):
    """Build a fully resolved :class:`fre.config.default.Config` for a domain."""
    overrides = config_overrides_from_args(args, domain)
    overrides.update(extra)
    try:
        return make_config(domain, **overrides)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("make_config failed (%s); falling back to Config(...)", exc)
        from fre.config.default import Config

        valid = {k: v for k, v in overrides.items() if k in set(dir(Config))}
        return Config(**valid)


def resolve_schedule(config: Any, args: argparse.Namespace) -> Tuple[int, int]:
    """Resolve the (encoder_steps, policy_steps) budgets for this run.

    Table 3: 150,000 encoder steps / 850,000 policy steps, and 1M / 1M for the
    ExORL and Kitchen domains.  CLI overrides win (useful for smoke tests).
    """
    encoder_steps = int(getattr(config, "encoder_train_steps", 150_000))
    policy_steps = int(getattr(config, "policy_train_steps", 850_000))
    try:
        encoder_steps = int(config.encoder_steps())
        policy_steps = int(config.policy_steps())
    except Exception:
        pass

    if args.encoder_steps is not None:
        encoder_steps = int(args.encoder_steps)
    if args.policy_steps is not None:
        policy_steps = int(args.policy_steps)
    if args.scale_steps is not None:
        encoder_steps = max(1, int(encoder_steps * float(args.scale_steps)))
        policy_steps = max(1, int(policy_steps * float(args.scale_steps)))
    return encoder_steps, policy_steps


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(config: Any, args: argparse.Namespace, domain: str):
    """Load (or synthesize) the unlabeled offline dataset for ``domain``."""
    from fre.envs.d4rl_loader import load_offline_dataset

    use_physics = None
    if args.no_physics:
        use_physics = False

    try:
        dataset = load_offline_dataset(
            config=config,
            domain=domain,
            dataset_path=args.dataset_path,
            dataset_dir=args.dataset_dir,
            variant=args.exorl_variant,
            use_physics=use_physics,
            allow_synthetic=bool(args.allow_synthetic),
            synthetic_transitions=int(args.synthetic_transitions),
        )
    except TypeError:
        # Older/newer loader signature: retry with the minimal keyword set.
        dataset = load_offline_dataset(
            config=config,
            domain=domain,
            allow_synthetic=bool(args.allow_synthetic),
        )

    if getattr(dataset, "synthetic", False):
        warnings.warn(
            f"[{domain}] using a SYNTHETIC dataset -- scores are not meaningful; "
            "pass --dataset-path to point at the real D4RL/ExORL data.",
            RuntimeWarning,
            stacklevel=2,
        )
    LOG.info(
        "[%s] dataset: %s transitions, obs_dim=%s, actions=%s, synthetic=%s",
        domain,
        getattr(dataset, "num_transitions", "?"),
        getattr(dataset, "obs_dim", "?"),
        getattr(dataset, "has_actions", "?"),
        getattr(dataset, "synthetic", False),
    )
    return dataset


def state_dims(config: Any, dataset: Any) -> Tuple[int, int]:
    """Return (encoder_state_dim, plain_obs_dim) for a dataset."""
    obs_dim = int(getattr(dataset, "obs_dim", 0) or 0)
    if obs_dim == 0:
        obs = _dataset_obs(dataset)
        if obs is not None:
            obs_dim = int(obs.shape[-1])
    encoder_dim = int(getattr(dataset, "encoder_state_dim", obs_dim) or obs_dim)
    return encoder_dim, obs_dim


# ---------------------------------------------------------------------------
# FRE training (Algorithm 1)
# ---------------------------------------------------------------------------


def build_fre_components(config: Any, dataset: Any, device: str):
    """Build the FRE model (+prior sampler) for a dataset.

    Returns ``(model, prior)`` where ``model`` is a
    :class:`fre.fre.fre_model.FREModel` and ``prior`` a
    :class:`fre.fre.prior.PriorSampler`.
    """
    from fre.fre.fre_model import FREModel
    from fre.fre.prior import make_prior_sampler

    encoder_dim, _ = state_dims(config, dataset)

    model = call_with_supported_kwargs(FREModel.from_config, config, encoder_dim)
    if model is None:  # pragma: no cover - defensive
        raise RuntimeError("FREModel.from_config returned None")

    prior_overrides: Dict[str, Any] = {}
    domain = str(getattr(config, "domain", "") or "")
    if domain.startswith("ant") and not getattr(config, "linear_exclude_dims", None):
        prior_overrides["exclude_dims"] = (0, 1)
    ratios = getattr(config, "prior_ratios", None)
    if ratios:
        prior_overrides["ratios"] = ratios
    prior = call_with_supported_kwargs(
        make_prior_sampler, config, encoder_dim, **prior_overrides
    )
    return model, prior


def make_fre_trainer(config: Any, dataset: Any, args: argparse.Namespace,
                     device: str, run_dir: str, logger: MetricLogger):
    """Instantiate :class:`fre.fre.trainer.FRETrainer` with paper hyperparameters."""
    from fre.fre.trainer import FRETrainer, make_trainer

    encoder_steps, policy_steps = resolve_schedule(config, args)
    _, obs_dim = state_dims(config, dataset)
    domain = str(getattr(config, "domain", ""))
    use_physics = bool(
        not args.no_physics and domain.startswith("exorl")
    )

    kwargs: Dict[str, Any] = dict(
        config=config,
        device=device,
        encoder_steps=encoder_steps,
        policy_steps=policy_steps,
        batch_size=int(getattr(config, "batch_size", 512)),
        learning_rate=float(getattr(config, "learning_rate", 1e-4)),
        grad_clip_norm=float(getattr(config, "grad_clip_norm", 10.0)),
        log_interval=int(args.log_interval),
        eval_interval=int(args.eval_interval),
        save_interval=int(args.save_interval),
        output_dir=run_dir,
        seed=int(args.seed),
        use_physics_augmentation=use_physics,
        logger=logger,
        beta_kl=getattr(config, "beta_kl", None),
    )
    try:
        trainer = make_trainer(config, dataset, **kwargs)
    except TypeError:
        trainer = FRETrainer(
            model=build_fre_components(config, dataset, device)[0],
            prior=build_fre_components(config, dataset, device)[1],
            dataset=dataset,
            **{k: v for k, v in kwargs.items() if k != "config"},
        )

    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            call_with_supported_kwargs(trainer.load, args.checkpoint, map_location=device)
            LOG.info("resumed checkpoint %s", args.checkpoint)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("could not load checkpoint %s (%s)", args.checkpoint, exc)
    return trainer, encoder_steps, policy_steps


def train_fre_domain(config: Any, dataset: Any, args: argparse.Namespace,
                     device: str, run_dir: str, logger: MetricLogger) -> Dict[str, Any]:
    """Run Algorithm 1 (phase 1 -> freeze -> phase 2) for one domain."""
    trainer, encoder_steps, policy_steps = make_fre_trainer(
        config, dataset, args, device, run_dir, logger
    )
    stage = "train" if args.stage == "all" else args.stage
    stats: Dict[str, Any] = {
        "encoder_steps": encoder_steps,
        "policy_steps": policy_steps,
    }

    started = time.time()
    if stage in ("encoder", "train"):
        LOG.info("[fre] phase 1: training encoder/decoder for %d steps", encoder_steps)
        try:
            enc_stats = call_with_supported_kwargs(
                trainer.train_encoder, steps=encoder_steps, log_interval=args.log_interval
            )
            stats["encoder_stats"] = _jsonable(enc_stats)
        except TypeError:
            enc_stats = trainer.train_encoder()
            stats["encoder_stats"] = _jsonable(enc_stats)
        try:
            trainer.freeze_encoder()
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("freeze_encoder failed (%s)", exc)

    if stage in ("policy", "train"):
        LOG.info("[fre] phase 2: training z-conditioned IQL for %d steps", policy_steps)
        # Encoder must be frozen before the policy phase starts.
        try:
            trainer.freeze_encoder()
        except Exception:
            pass
        try:
            pol_stats = call_with_supported_kwargs(
                trainer.train_policy, steps=policy_steps, log_interval=args.log_interval
            )
            stats["policy_stats"] = _jsonable(pol_stats)
        except TypeError:
            pol_stats = trainer.train_policy()
            stats["policy_stats"] = _jsonable(pol_stats)

    stats["seconds"] = time.time() - started

    ckpt = args.save_checkpoint or os.path.join(run_dir, "ckpt.pt")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
        call_with_supported_kwargs(trainer.save, ckpt)
        stats["checkpoint"] = ckpt
        LOG.info("saved checkpoint %s", ckpt)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("could not save checkpoint (%s)", exc)

    stats["trainer"] = trainer
    return stats


def _jsonable(obj: Any) -> Any:
    """Convert dataclasses / numpy- / torch-scalars into JSON-friendly values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "as_dict"):
        try:
            return _jsonable(obj.as_dict())
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


# ---------------------------------------------------------------------------
# Zero-shot encoding of downstream tasks (32 reward-annotated samples)
# ---------------------------------------------------------------------------


def build_encoder_context(task: Any, dataset: Any, num_samples: int,
                          rng: Any, use_physics: bool = False) -> Tuple[Any, Any]:
    """Return ``(states, rewards)`` with K reward-annotated samples for ``task``.

    Implements the paper's evaluation protocol: a *new* task is specified by K =
    32 (state, reward) pairs obtained from the offline dataset, and the encoder
    maps them to z (contrast: FB/SF use 5120 samples).
    """
    # 1) Task object may know how to build its own context (e.g. AntMaze goals
    #    force the goal state into the context).
    encoder_pairs = getattr(task, "encoder_pairs", None)
    if callable(encoder_pairs):
        for kwargs in (
            dict(num_samples=num_samples, rng=rng),
            dict(num_samples=num_samples),
            dict(),
        ):
            try:
                states, rewards = encoder_pairs(dataset, **kwargs)
                return states, rewards
            except TypeError:
                continue
            except Exception:
                break

    # 2) Module-level helper from the environment suite that defines the task.
    module = sys.modules.get(type(task).__module__)
    helper = getattr(module, "sample_task_encoder_pairs", None) if module else None
    if callable(helper):
        for kwargs in (
            dict(num_samples=num_samples, rng=rng, use_physics=use_physics),
            dict(num_samples=num_samples, rng=rng),
            dict(num_samples=num_samples),
            dict(),
        ):
            try:
                out = helper(task, dataset, **kwargs)
                if isinstance(out, tuple) and len(out) == 2:
                    return out[0], out[1]
            except TypeError:
                continue
            except Exception:
                break

    # 3) Generic fallback: sample states from the dataset and evaluate the
    #    task's reward function on them.
    import numpy as np

    states = None
    if dataset is not None and hasattr(dataset, "sample_states"):
        try:
            states = np.asarray(dataset.sample_states(num_samples))
        except Exception:
            states = None
    if states is None:
        obs = _dataset_obs(dataset)
        if obs is None:
            raise RuntimeError("cannot build an encoder context without a dataset")
        obs = np.asarray(obs)
        idx = rng.integers(0, len(obs), size=min(num_samples, len(obs)))
        states = obs[idx]

    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None]
    states = states[:num_samples]

    reward_fn = getattr(task, "reward_from_state", None)
    if callable(reward_fn):
        try:
            rewards = np.asarray(reward_fn(states), dtype=np.float32).reshape(-1)
        except Exception:
            rewards = np.zeros(len(states), dtype=np.float32)
    else:
        rewards = np.zeros(len(states), dtype=np.float32)

    return states, rewards


def encode_task_z(encoder: Any, states: Any, rewards: Any, device: str,
                  deterministic: bool = True) -> Any:
    """Encode ``(states, rewards)`` into the 128-dim task latent z."""
    import numpy as np
    import torch

    s = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
    r = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=device)
    if s.dim() == 2:
        s = s.unsqueeze(0)
    if r.dim() == 1:
        r = r.unsqueeze(0)

    encoder.eval()
    with torch.no_grad():
        encode = getattr(encoder, "encode", None)
        if callable(encode):
            try:
                z = encode(s, r, sample=not deterministic)
            except TypeError:
                z = encode(s, r)
        else:
            dist = encoder(s, r)
            z = dist.mean if deterministic else dist.rsample()
    return z


def make_policy_fn(agent: Any, z: Any, device: str, deterministic: bool = True) -> Callable:
    """Wrap a z-conditioned agent into ``act_fn(obs) -> action`` for rollouts."""
    import numpy as np

    z_np = None
    try:
        z_np = z.detach().cpu().numpy()
    except AttributeError:
        z_np = np.asarray(z)

    def act_fn(obs):  # noqa: ANN001
        select = getattr(agent, "select_action", None) or getattr(agent, "act", None)
        if select is None:
            raise RuntimeError("agent exposes neither select_action() nor act()")
        try:
            action = select(obs, z_np, deterministic=deterministic)
        except TypeError:
            action = select(obs, z_np)
        return np.asarray(action)

    return act_fn


# ---------------------------------------------------------------------------
# Evaluation drivers (per domain)
# ---------------------------------------------------------------------------


def evaluate_antmaze(agent: Any, encoder: Any, dataset: Any, config: Any,
                     args: argparse.Namespace, device: str, seed: int,
                     deterministic: bool) -> Dict[str, Any]:
    """Evaluate the AntMaze suite (goal-reaching / directional / simplex / paths)."""
    import numpy as np

    from fre.envs.antmaze_eval import (
        evaluate_task,
        make_antmaze_task_suite,
    )

    num_samples = int(args.eval_samples or getattr(config, "fre_eval_samples", 32))
    rng = np.random.default_rng(seed)
    tasks = _flatten_tasks(make_antmaze_task_suite())
    suites = set(args.suites) if args.suites else None

    results: Dict[str, Any] = {}
    for name, task in tasks:
        if suites and not any(s in name for s in suites):
            continue
        states, rewards = build_encoder_context(task, dataset, num_samples, rng)
        if encoder is not None:
            z = encode_task_z(encoder, states, rewards, device, deterministic)
        else:
            # Baselines (GC-IQL / GC-BC) receive the ground-truth goal directly.
            z = _gc_goal_vector(task, config)
            if z is None:
                LOG.info("[antmaze] skipping %s (no goal representation for this baseline)", name)
                continue
        act_fn = make_policy_fn(agent, z, device, deterministic)
        results[name] = call_with_supported_kwargs(
            evaluate_task,
            task,
            act_fn,
            num_episodes=int(args.num_eval_episodes or getattr(config, "num_eval_episodes", 20)),
            max_episode_steps=int(getattr(config, "antmaze_max_episode_steps", 2000)),
            seed=seed,
            discretize_xy=bool(getattr(config, "antmaze_discretize_xy", True)),
        )
    return results


def _gc_goal_vector(task: Any, config: Any) -> Optional[Any]:
    """Goal vector for goal-conditioned baselines (None for non-goal tasks)."""
    import numpy as np

    goal = getattr(task, "goal", None)
    if goal is None:
        goal = getattr(task, "goal_position", None)
        if callable(goal):
            try:
                goal = goal()
            except Exception:
                goal = None
    if goal is None:
        return None
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    dim = int(getattr(config, "goal_dim", goal.size) or goal.size)
    if goal.size < dim:
        goal = np.pad(goal, (0, dim - goal.size))
    return goal[:dim]


def evaluate_exorl(agent: Any, encoder: Any, dataset: Any, config: Any,
                   args: argparse.Namespace, device: str, seed: int,
                   deterministic: bool) -> Dict[str, Any]:
    """Evaluate the ExORL suite (walker/cheetah goals + velocity)."""
    import numpy as np

    from fre.envs.exorl_eval import (
        evaluate_task,
        make_goal_tasks,
        make_velocity_tasks,
    )

    domain = str(getattr(config, "domain", "exorl_walker"))
    num_samples = int(args.eval_samples or getattr(config, "fre_eval_samples", 32))
    rng = np.random.default_rng(seed)

    tasks: List[Any] = []
    try:
        tasks.extend(
            call_with_supported_kwargs(
                make_goal_tasks,
                domain,
                dataset,
                num_goals=5,
                seed=seed,
                max_episode_steps=int(getattr(config, "exorl_max_episode_steps", 1000)),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("[%s] could not build goal tasks (%s)", domain, exc)
    try:
        tasks.extend(
            call_with_supported_kwargs(
                make_velocity_tasks,
                domain=domain,
                max_episode_steps=int(getattr(config, "exorl_max_episode_steps", 1000)),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("[%s] could not build velocity tasks (%s)", domain, exc)

    use_physics = bool(not args.no_physics)
    results: Dict[str, Any] = {}
    for task in tasks:
        name = str(getattr(task, "name", None) or type(task).__name__)
        if args.suites and not any(s in name for s in args.suites):
            continue
        states, rewards = build_encoder_context(
            task, dataset, num_samples, rng, use_physics=use_physics
        )
        if encoder is not None:
            z = encode_task_z(encoder, states, rewards, device, deterministic)
        else:
            z = _exorl_baseline_latent(args.agent, task, dataset, config, num_samples)
            if z is None:
                LOG.info("[%s] skipping %s for agent %s", domain, name, args.agent)
                continue
        act_fn = make_policy_fn(agent, z, device, deterministic)
        results[name] = call_with_supported_kwargs(
            evaluate_task,
            task,
            act_fn,
            domain=domain,
            num_episodes=int(args.num_eval_episodes or getattr(config, "num_eval_episodes", 20)),
            max_episode_steps=int(getattr(config, "exorl_max_episode_steps", 1000)),
            seed=seed,
            use_physics=use_physics,
        )
    return results


def _exorl_baseline_latent(agent_name: str, task: Any, dataset: Any, config: Any,
                           num_samples: int) -> Optional[Any]:
    """Latent for goal-conditioned baselines on ExORL (goal state or None)."""
    import numpy as np

    goal = getattr(task, "goal_state", None)
    if goal is None:
        goal = getattr(task, "goal", None)
    if goal is None:
        return None
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    dim = int(getattr(config, "goal_dim", goal.size) or goal.size)
    if goal.size < dim:
        goal = np.pad(goal, (0, dim - goal.size))
    return goal[:dim]


def evaluate_kitchen(agent: Any, encoder: Any, dataset: Any, config: Any,
                     args: argparse.Namespace, device: str, seed: int,
                     deterministic: bool) -> Dict[str, Any]:
    """Evaluate the 7 D4RL Kitchen subtasks (sparse rewards)."""
    import numpy as np

    from fre.envs.kitchen_eval import evaluate_task, get_task_suite

    num_samples = int(args.eval_samples or getattr(config, "fre_eval_samples", 32))
    rng = np.random.default_rng(seed)
    suite = get_task_suite("kitchen")
    tasks = _flatten_tasks(suite)

    results: Dict[str, Any] = {}
    for name, task in tasks:
        if args.suites and not any(s in name for s in args.suites):
            continue
        states, rewards = build_encoder_context(task, dataset, num_samples, rng)
        if encoder is not None:
            z = encode_task_z(encoder, states, rewards, device, deterministic)
        else:
            LOG.info("[kitchen] skipping %s for non-FRE agent %s", name, args.agent)
            continue
        act_fn = make_policy_fn(agent, z, device, deterministic)
        results[name] = call_with_supported_kwargs(
            evaluate_task,
            task,
            act_fn,
            env_id=str(getattr(config, "env_id", "kitchen-complete-v0")),
            num_episodes=int(args.num_eval_episodes or getattr(config, "num_eval_episodes", 20)),
            max_episode_steps=int(getattr(config, "kitchen_max_episode_steps", 1000)),
            seed=seed,
        )
    return results


DOMAIN_EVALUATORS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "antmaze": evaluate_antmaze,
    "exorl_walker": evaluate_exorl,
    "exorl_cheetah": evaluate_exorl,
    "kitchen": evaluate_kitchen,
}


def domain_row_name(domain: str) -> str:
    """Table 1 aggregate row name for a domain (``antmaze-all`` / ``kitchen``)."""
    if domain == "kitchen":
        return "kitchen"
    return f"{domain.replace('_', '-')}-all"


# ---------------------------------------------------------------------------
# Baseline training/eval glue
# ---------------------------------------------------------------------------


def train_agent_domain(config: Any, dataset: Any, args: argparse.Namespace,
                       device: str, run_dir: str, logger: MetricLogger
                       ) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """Train the requested agent; return ``(agent, encoder, stats)``.

    ``encoder`` is the frozen FRE encoder for FRE (used for zero-shot encoding)
    and ``None`` for the baselines that receive the task through another route.
    """
    agent_name = args.agent
    _, obs_dim = state_dims(config, dataset)
    action_dim = int(getattr(dataset, "action_dim", 0) or 1)
    stats: Dict[str, Any] = {"agent": agent_name}

    if agent_name == "fre":
        stats.update(train_fre_domain(config, dataset, args, device, run_dir, logger))
        trainer = stats.pop("trainer", None)
        encoder = getattr(getattr(trainer, "model", None), "encoder", None)
        agent = getattr(trainer, "iql", None) or getattr(trainer, "agent", None)
        return agent, encoder, stats

    encoder_steps, policy_steps = resolve_schedule(config, args)

    if agent_name in ("gc_iql", "gc_bc"):
        from fre.baselines import gc_bc as gc_bc_mod
        from fre.baselines import gc_iql as gc_iql_mod

        if agent_name == "gc_iql":
            make_fn, train_fn = gc_iql_mod.make_gc_iql, gc_iql_mod.train_gc_iql
        else:
            make_fn, train_fn = gc_bc_mod.make_gc_bc, gc_bc_mod.train_gc_bc

        agent = call_with_supported_kwargs(make_fn, config, obs_dim, action_dim, device=device)
        try:
            agent, history = call_with_supported_kwargs(
                train_fn,
                config,
                dataset,
                obs_dim,
                action_dim,
                steps=int(policy_steps),
                batch_size=int(getattr(config, "batch_size", 512)),
                log_interval=int(args.log_interval),
                logger=logger,
                device=device,
            )
        except TypeError:  # pragma: no cover - defensive
            agent, history = train_fn(config, dataset, obs_dim, action_dim)
        stats["history"] = _jsonable(history)[-50:] if isinstance(history, list) else history
        return agent, None, stats

    if agent_name == "opal":
        from fre.baselines import opal as opal_mod

        agent = opal_mod.make_opal(config, obs_dim, action_dim, device=device)
        try:
            agent, history = call_with_supported_kwargs(
                opal_mod.train_opal,
                config,
                dataset,
                obs_dim,
                action_dim,
                skill_steps=int(encoder_steps),
                policy_steps=int(policy_steps),
                batch_size=int(getattr(config, "batch_size", 512)),
                log_interval=int(args.log_interval),
                logger=logger,
                agent=agent,
                device=device,
            )
        except TypeError:  # pragma: no cover - defensive
            agent, history = opal_mod.train_opal(
                config, dataset, obs_dim=obs_dim, action_dim=action_dim
            )
        stats["history"] = _jsonable(history) if not isinstance(history, list) else _jsonable(history)[-50:]
        return agent, None, stats

    if agent_name in ("fb", "sf"):
        if agent_name == "fb":
            from fre.baselines import forward_backward as fb_mod

            make_fn, train_fn = fb_mod.make_forward_backward, fb_mod.train_forward_backward
        else:
            from fre.baselines import successor_features as sf_mod

            make_fn, train_fn = sf_mod.make_successor_features, sf_mod.train_successor_features

        agent = call_with_supported_kwargs(make_fn, config, obs_dim, action_dim, device=device)
        try:
            agent, history = call_with_supported_kwargs(
                train_fn,
                config,
                dataset,
                obs_dim=obs_dim,
                action_dim=action_dim,
                steps=int(policy_steps),
                log_interval=int(args.log_interval),
                logger=logger,
                agent=agent,
                device=device,
            )
        except TypeError:  # pragma: no cover - defensive
            agent, history = train_fn(config, dataset)
        stats["history"] = _jsonable(history) if not isinstance(history, list) else _jsonable(history)[-50:]

        # FB / SF need 5120 reward samples at evaluation (Sec. 5.2).
        stats["eval_samples"] = 5120
        if args.eval_samples is None:
            args.eval_samples = 5120
        return agent, None, stats

    raise ValueError(f"unsupported agent {agent_name!r}")


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------


def aggregate_results(per_seed: Dict[int, Dict[str, Any]], domain: str) -> Dict[str, Any]:
    """Aggregate per-seed task results into mean/std rows (Table 1 protocol)."""
    acc = ResultAccumulator(name=domain, ddof=1)
    for seed, results in per_seed.items():
        filtered = {k: v for k, v in results.items() if k != domain_row_name(domain)}
        acc.add_seed(int(seed), filtered)
    table = acc.table()
    return {"per_seed": per_seed, "table": table, "accumulator": acc}


def print_report(agent: str, domains: List[str], reports: Dict[str, Any],
                 out: Optional[Any] = None) -> None:
    """Print a Table-1-style report with published reference numbers."""
    stream = out or sys.stdout
    header = f"\n{'=' * 78}\nFRE reproduction report -- agent: {agent}\n{'=' * 78}"
    print(header, file=stream)

    for domain in domains:
        report = reports.get(domain, {})
        table = report.get("table") or {}
        if not table:
            continue
        print(f"\n[{domain}]", file=stream)
        for row, values in table.items():
            mean = float(values.get("mean", float("nan")))
            std = float(values.get("std", 0.0))
            ref = lookup_reference(agent, row)
            if ref is not None:
                delta = mean - ref[0]
                flag = "ok" if abs(delta) <= max(1.0, ref[1]) else "  "
                print(
                    f"  {row:<28} {mean:6.1f} +- {std:4.1f}   "
                    f"paper {ref[0]:6.1f} +- {ref[1]:4.1f}  d={delta:+6.1f} {flag}",
                    file=stream,
                )
            else:
                print(f"  {row:<28} {mean:6.1f} +- {std:4.1f}", file=stream)

    print(f"\n{'=' * 78}\n", file=stream)


def print_plan(domains: List[str], args: argparse.Namespace) -> None:
    """``--dry-run``: show the fully resolved configuration without training."""
    print("FRE dry run")
    print("-" * 60)
    print(f"  agent            : {args.agent}")
    print(f"  stage            : {args.stage}")
    print(f"  domains          : {', '.join(domains)}")
    print(f"  seeds            : {args.seeds or list(range(args.num_eval_seeds or 5))}")
    print(f"  eval episodes    : {args.num_eval_episodes or 'config default (20)'}")
    print(f"  prior mixture    : {args.prior_mixture} -> {PRIOR_MIXTURES[args.prior_mixture]}")
    print(f"  device           : {args.device or 'auto'}")
    print(f"  output dir       : {args.output_dir or 'runs'}")
    for domain in domains:
        config = build_config(domain, args)
        enc_steps, pol_steps = resolve_schedule(config, args)
        print(f"\n  [{domain}]")
        print(f"    env_id          : {getattr(config, 'env_id', '?')}")
        print(f"    dataset_id      : {getattr(config, 'dataset_id', '?')}")
        print(f"    encoder steps   : {enc_steps}")
        print(f"    policy steps    : {pol_steps}")
        print(f"    K / K'          : {getattr(config, 'num_encoder_samples', '?')} / "
              f"{getattr(config, 'num_decoder_samples', '?')}")
        print(f"    prior ratios    : {getattr(config, 'prior_ratios', '?')}")
        print(f"    max ep steps    : {getattr(config, 'max_episode_steps', '?')}")
        try:
            tasks = get_eval_tasks(domain)
            print(f"    eval tasks      : {len(tasks)}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_domain(domain: str, args: argparse.Namespace, run_dir: str) -> Dict[str, Any]:
    """Train (if requested) and evaluate one domain for one agent."""
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = build_config(domain, args)
    logger = MetricLogger(
        log_dir=os.path.join(run_dir, domain),
        name=f"{args.agent}-{domain}",
        use_tensorboard=bool(args.tensorboard),
    )

    report: Dict[str, Any] = {"domain": domain, "agent": args.agent, "device": device}
    logger.log_hyperparameters(
        {k: v for k, v in config.to_dict().items()
         if isinstance(v, (int, float, str, bool, tuple))}
    )

    if args.stage == "eval" and args.checkpoint is None:
        raise SystemExit(
            "--stage eval requires --checkpoint (no training data would be used)."
        )

    # --- dataset -----------------------------------------------------------
    dataset = None
    if args.stage != "eval":
        dataset = load_dataset(config, args, domain)
    else:
        try:
            dataset = load_dataset(config, args, domain)
        except Exception as exc:
            LOG.warning("[%s] no dataset available for evaluation contexts (%s)", domain, exc)

    # --- training ----------------------------------------------------------
    agent = encoder = None
    if args.stage != "eval":
        agent, encoder, train_stats = train_agent_domain(
            config, dataset, args, device, run_dir, logger
        )
        report["training"] = {k: v for k, v in train_stats.items() if k != "trainer"}
        report["trainer"] = train_stats.get("trainer")
    elif args.checkpoint:
        # Rebuild the model/agent topology, then load the checkpoint.
        agent, encoder, eval_stats = _load_checkpoint_for_eval(
            config, dataset, args, device, logger
        )
        report["training"] = eval_stats

    # --- evaluation --------------------------------------------------------
    if not args.no_eval and dataset is not None:
        per_seed: Dict[int, Dict[str, Any]] = {}
        seeds = args.seeds or list(range(int(args.num_eval_seeds or 5)))
        evaluator = DOMAIN_EVALUATORS.get(domain)
        if evaluator is None:
            LOG.warning("no evaluator for domain %s", domain)
        else:
            for seed in seeds:
                set_global_seed(int(seed))
                try:
                    results = evaluator(
                        agent, encoder, dataset, config, args, device, int(seed),
                        bool(args.deterministic_eval),
                    )
                except Exception as exc:  # pragma: no cover - env availability
                    LOG.warning("[%s] evaluation failed for seed %d: %s", domain, seed, exc)
                    LOG.debug("%s", traceback.format_exc())
                    continue
                per_seed[int(seed)] = results
                logger.set_step(int(seed))
                logger.log({f"eval/{k}": _score_of(v) for k, v in results.items()})

            if per_seed:
                aggregated = aggregate_results(per_seed, domain)
                report["eval"] = {
                    "per_seed": {k: _jsonable(v) for k, v in per_seed.items()},
                    "table": _jsonable(aggregated["table"]),
                }
                report["accumulator"] = aggregated["accumulator"]

    if hasattr(agent, "select_action"):  # keep torch happy on exit
        pass
    logger.close()
    return report


def _score_of(value: Any) -> float:
    """Extract the scalar score from an evaluation-result payload."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "normalized_return", "mean", "total_return"):
            if key in value:
                try:
                    return float(value[key])
                except (TypeError, ValueError):
                    return float("nan")
    if hasattr(value, "normalized_return"):
        try:
            return float(value.normalized_return)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "as_dict"):
        try:
            return _score_of(value.as_dict())
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_checkpoint_for_eval(config: Any, dataset: Any, args: argparse.Namespace,
                              device: str, logger: MetricLogger):
    """Rebuild the FRE model topology and load ``--checkpoint`` for evaluation."""
    from fre.fre.trainer import FRETrainer
    from fre.fre.trainer import make_trainer as _make_trainer

    model, prior = build_fre_components(config, dataset, device)
    trainer = None
    try:
        trainer = call_with_supported_kwargs(
            _make_trainer,
            config,
            dataset,
            device=device,
            logger=logger,
            output_dir=os.path.dirname(os.path.abspath(args.checkpoint)),
        )
    except Exception:
        try:
            trainer = FRETrainer(model=model, prior=prior, dataset=dataset, device=device)
        except Exception:
            trainer = None

    if trainer is not None:
        try:
            call_with_supported_kwargs(trainer.load, args.checkpoint, map_location=device)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("failed to load %s (%s)", args.checkpoint, exc)
        try:
            trainer.freeze_encoder()
        except Exception:
            pass
        encoder = getattr(getattr(trainer, "model", None), "encoder", None) or model.encoder
        agent = getattr(trainer, "iql", None) or getattr(trainer, "agent", None)
        return agent, encoder, {"checkpoint": args.checkpoint, "encoder_steps": 0, "policy_steps": 0}

    LOG.warning("could not build a trainer for evaluation; encoder-only evaluation")
    return None, model.encoder, {"checkpoint": args.checkpoint, "encoder_steps": 0, "policy_steps": 0}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        domains = resolve_domains(args.domain)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run or args.stage == "dry-run":
        print_plan(domains, args)
        return 0

    tag = args.tag or f"{args.agent}-{args.domain}"
    run_dir = args.output_dir or make_run_dir(
        root="runs", name=tag, seed=args.seed, subdirs=["logs"]
    )
    os.makedirs(run_dir, exist_ok=True)
    LOG.info("run directory: %s", run_dir)

    reports: Dict[str, Any] = {}
    overall_start = time.time()
    for domain in domains:
        try:
            reports[domain] = run_domain(domain, args, run_dir)
        except SystemExit:
            raise
        except Exception as exc:  # pragma: no cover - keep multi-domain runs alive
            LOG.error("[%s] run failed: %s", domain, exc)
            LOG.debug("%s", traceback.format_exc())
            reports[domain] = {"domain": domain, "error": str(exc)}
        finally:
            # Drop non-serializable objects before reporting.
            for report in reports.values():
                report.pop("trainer", None)
                report.pop("accumulator", None)

    if not args.no_eval:
        print_report(args.agent, domains, reports)

    payload = {
        "agent": args.agent,
        "stage": args.stage,
        "domains": domains,
        "prior_mixture": args.prior_mixture,
        "seeds": args.seeds or list(range(int(args.num_eval_seeds or 5))),
        "reports": _jsonable(reports),
        "seconds": time.time() - overall_start,
        "paper_reference": {
            "table1": _jsonable(TABLE1_REFERENCE),
            "fre_targets": _jsonable(TABLE1_FRE_TARGETS),
            "aggregate": _jsonable(TABLE1_AGGREGATE_TARGETS),
        },
    }
    out_path = args.json or os.path.join(run_dir, "report.json")
    try:
        save_json(out_path, payload)
        LOG.info("wrote report to %s", out_path)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("could not write report (%s)", exc)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
