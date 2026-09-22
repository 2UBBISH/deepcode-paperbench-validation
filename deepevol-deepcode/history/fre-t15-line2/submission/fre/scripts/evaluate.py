#!/usr/bin/env python
"""Zero-shot evaluation entry point for Free-form Reward Encodings (FRE).

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings".

Evaluation protocol (Source: Section 5.2, verbatim):
    "All methods are evaluated using a mean over twenty evaluation episodes, and each
     agent is trained using five random seeds, with the standard deviation across seeds
     shown. FRE, GC-IQL, and GC-BC are implemented within the same codebase and with the
     same network structure."

At test time FRE encodes a small set of reward-annotated states of the *novel* task
(32 ``(s, eta(s))`` pairs, per Table 1's caption) into a latent ``z ~ p_theta(z | .)`` and
rolls out the frozen ``pi(a | s, z)`` policy with no further training whatsoever.  The
benchmark suites (AntMaze / ExORL / Kitchen) are built by the modules in ``fre.envs``; the
rollout + scoring machinery lives in ``fre.eval.zero_shot_eval`` (for FRE) and
``fre.eval.baselines`` (for GC-BC / GC-IQL / OPAL-10).  Episode returns are normalized to
the paper's 0-100 scale (Source: Table 1 caption).

Usage examples
--------------
FRE on AntMaze (Table 1 main result)::

    python scripts/evaluate.py --config configs/fre_antmaze.yaml \\
        --domain antmaze --method FRE \\
        --checkpoint runs/antmaze/encoder.pt --policy runs/antmaze/policy.pt \\
        --suites ant-goal-reaching ant-directional ant-random-simplex \\
                 ant-path-center ant-path-loop ant-path-edges \\
        --seeds 0 1 2 3 4 --num-episodes 20 --save results/fre_antmaze.json

FRE on ExORL walker + cheetah (Source: Appendix C.2, RND datasets)::

    python scripts/evaluate.py --config configs/fre_exorl.yaml \\
        --domain exorl --dataset walker --method FRE \\
        --checkpoint runs/exorl_walker/encoder.pt --policy runs/exorl_walker/policy.pt \\
        --suites exorl-walker-velocity exorl-walker-goals

Baselines (interfaces are identical across methods)::

    python scripts/evaluate.py --config configs/fre_antmaze.yaml \\
        --domain antmaze --method GC-IQL --policy runs/gciql/agent.pt
    python scripts/evaluate.py --config configs/fre_antmaze.yaml \\
        --domain antmaze --method GC-BC  --policy runs/gcbc/agent.pt
    python scripts/evaluate.py --config configs/fre_exorl.yaml \\
        --domain exorl --dataset walker --method OPAL-10 --policy runs/opal/agent.pt

Configuration values that must be exact (Source: Table 3 / Section 5.2): 20 evaluation
episodes, 5 training seeds, 32 encoded reward pairs for FRE (5120 for FB/SF), returns
normalized into ``[0, 100]``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Package import plumbing: allow `python scripts/evaluate.py` from the repository root.
# --------------------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # .../fre
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --------------------------------------------------------------------------------------
# Protocol constants (Source: Section 5.2 and Table 1 caption)
# --------------------------------------------------------------------------------------
NUM_EVAL_EPISODES = 20          # "a mean over twenty evaluation episodes"
NUM_TRAINING_SEEDS = 5          # "each agent is trained using five random seeds"
FRE_CONTEXT_SAMPLES = 32        # FRE encodes 32 (state, reward) pairs at test time
FB_SF_CONTEXT_SAMPLES = 5120    # FB / SF use 5120 (Table 1 caption)
NORMALIZED_RETURN_MIN = 0.0
NORMALIZED_RETURN_MAX = 100.0

#: Suite groups exposed by the domain task modules, keyed by (domain, config dataset).
DOMAIN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "antmaze": {
        "env_id": "antmaze-large-diverse-v2",
        "suites": [
            "ant-goal-reaching",
            "ant-directional",
            "ant-random-simplex",
            "ant-path-center",
            "ant-path-loop",
            "ant-path-edges",
        ],
        "episode_length": 2000,   # Source: Appendix C.1 "length of 2000 timesteps"
        "start_at_center": True,  # Source: Appendix C.1 ant placed at maze center
    },
    "exorl": {
        "env_id": "walker-run",
        "suites": ["exorl-walker-goals", "exorl-walker-velocity"],
        "episode_length": 1000,   # Source: Appendix C.2 "evaluated for 1000 timesteps"
        "start_at_center": False,
    },
    "kitchen": {
        "env_id": "kitchen-complete-v0",
        "suites": ["kitchen"],
        # NOTE (Source: not specified in the paper) Kitchen episode length defaults to
        # the D4RL convention of 1000 steps.
        "episode_length": 1000,
        "start_at_center": False,
    },
}

#: Method names understood by this script.
METHODS = ("FRE", "GC-IQL", "GC-BC", "OPAL-10", "OPAL")


# ======================================================================================
# Small utilities
# ======================================================================================
def filter_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keyword arguments ``fn`` actually accepts.

    The training/eval code in this repository supports several interchangeable
    collaborators (rollout functions, env factories, loggers), each with slightly
    different signatures.  Rather than coupling this script to one exact signature we
    pass everything that might be useful and drop what is not accepted.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def call_flexibly(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments it accepts."""
    return fn(*args, **filter_kwargs(fn, kwargs))


def resolve_device(requested: Optional[str]) -> str:
    """Resolve a device string, degrading to CPU when CUDA is unavailable."""
    if not requested:
        requested = "cuda" if _torch_cuda_available() else "cpu"
    if str(requested).startswith("cuda") and not _torch_cuda_available():
        return "cpu"
    return str(requested)


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch absent
        return False


def torch():
    """Lazily import torch with a helpful error message (keeps --help cheap)."""
    try:
        import torch as _torch

        return _torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required to run zero-shot evaluation "
            "(models are torch modules). Please install the CUDA/CPU build of torch."
        ) from exc


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML (or JSON) config file; returns ``{}`` when ``path`` is None."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r") as handle:
        text = handle.read()
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except ImportError:  # pragma: no cover - fallback when PyYAML is unavailable
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ImportError(
                "PyYAML is required to read YAML configs; alternatively pass a JSON "
                f"config file. Original error: {exc}"
            ) from exc
    return loaded or {}


def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``cfg['a']['b']`` via the dotted key ``"a.b"``."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def resolve_suites(domain: str, requested: Optional[Sequence[str]],
                   config: Optional[Sequence[str]] = None) -> List[str]:
    """Pick the suite groups to evaluate (CLI > config > domain default)."""
    names = list(requested) if requested else []
    if not names and config:
        names = list(config)
    if not names:
        names = list(DOMAIN_DEFAULTS.get(domain, {}).get("suites", []))
    return names


def ensure_sequence(value: Any, default: Sequence[int] = ()) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(v) for v in value]


# ======================================================================================
# Dataset / model construction
# ======================================================================================
def build_dataset(domain: str, cfg: Dict[str, Any], args: argparse.Namespace) -> Any:
    """Instantiate the offline dataset wrapper (AntMaze / ExORL / Kitchen).

    Each loader returns an object exposing ``replay_buffer``, ``obs_dim``,
    ``encoder_obs_dim`` and the ``sample_states`` / ``sample_states_with_metadata``
    interface required by the reward priors and the evaluator.
    """
    dataset_cfg = cfg.get("dataset", {}) or {}
    seed = args.seed
    domain = domain.lower()

    if domain == "antmaze":
        from fre.data.antmaze_dataset import load_antmaze_dataset

        return call_flexibly(
            load_antmaze_dataset,
            path=first_not_none(args.dataset, dataset_cfg.get("path")),
            dataset_name=first_not_none(
                dataset_cfg.get("name"),
                dataset_cfg.get("env_id"),
                DOMAIN_DEFAULTS["antmaze"]["env_id"],
            ),
            env_id=dataset_cfg.get("env_id"),
            num_bins=int(dataset_cfg.get("num_xy_bins", 32)),
            discretize=bool(dataset_cfg.get("discretize", True)),
            discretize_mode=dataset_cfg.get("discretize_mode", "index"),
            bounds=_as_bounds(dataset_cfg.get("xy_bounds")),
            build_buffer=True,
            exclude_final_states=bool(dataset_cfg.get("exclude_final_states", True)),
            attach_stats=bool(dataset_cfg.get("attach_stats", True)),
            limit=dataset_cfg.get("limit"),
            seed=seed,
        )

    if domain == "exorl":
        from fre.data.exorl_dataset import load_exorl_dataset

        domain_name = first_not_none(args.dataset, dataset_cfg.get("domain"), "walker")
        task = first_not_none(args.task, dataset_cfg.get("task"))
        dataset_name = first_not_none(
            dataset_cfg.get("name"), (f"{domain_name}-{task}" if task else None)
        )
        return call_flexibly(
            load_exorl_dataset,
            path=first_not_none(args.dataset_dir, dataset_cfg.get("path")),
            domain=domain_name,
            task=task,
            dataset_name=dataset_name,
            dataset=str(dataset_cfg.get("dataset", "rnd")),  # RND exploratory datasets
            physics_method=cfg_get(cfg, "dataset.physics.method", "auto"),
            normalize=bool(dataset_cfg.get("normalize", True)),
            select_goals=True,
            num_goals=int(cfg_get(cfg, "evaluation.num_goals", 5)),
            build_buffer=True,
            exclude_final_states=bool(dataset_cfg.get("exclude_final_states", True)),
            attach_stats=bool(dataset_cfg.get("attach_stats", True)),
            max_transitions=dataset_cfg.get("limit"),
            seed=seed,
        )

    if domain == "kitchen":
        from fre.data.kitchen_dataset import KITCHEN_TASKS, load_kitchen_dataset

        return call_flexibly(
            load_kitchen_dataset,
            path=first_not_none(args.dataset, dataset_cfg.get("path")),
            dataset_name=first_not_none(
                dataset_cfg.get("name"),
                dataset_cfg.get("env_id"),
                DOMAIN_DEFAULTS["kitchen"]["env_id"],
            ),
            env_id=dataset_cfg.get("env_id"),
            tasks=KITCHEN_TASKS,
            task_threshold=float(cfg_get(cfg, "evaluation.task_kwargs.threshold", 0.3)),
            eval_episode_length=int(
                cfg_get(cfg, "evaluation.eval_episode_length",
                        DOMAIN_DEFAULTS["kitchen"]["episode_length"])
            ),
            build_buffer=True,
            exclude_final_states=bool(dataset_cfg.get("exclude_final_states", True)),
            attach_stats=bool(dataset_cfg.get("attach_stats", True)),
            limit=dataset_cfg.get("limit"),
            seed=seed,
        )

    raise ValueError(f"unknown domain '{domain}' (expected antmaze/exorl/kitchen)")


def _as_bounds(bounds: Any) -> Any:
    """Coerce YAML list-of-lists bounds into a tuple of tuples."""
    if bounds is None:
        return None
    try:
        return tuple(tuple(float(v) for v in pair) for pair in bounds)
    except TypeError:
        return None


def build_encoder(domain: str, dataset: Any, cfg: Dict[str, Any], args: argparse.Namespace,
                  device: str) -> Any:
    """Build the FRE encoder with the paper's Table 3 architecture."""
    from fre.models.encoder import make_fre_encoder

    model_cfg = cfg.get("model", {}) or {}
    encoder_input_dim = int(
        first_not_none(
            getattr(dataset, "encoder_obs_dim", None),
            getattr(dataset, "obs_dim", None),
            cfg_get(cfg, "dataset.obs_dim"),
            _infer_obs_dim(dataset),
        )
    )
    encoder = call_flexibly(
        make_fre_encoder,
        state_dim=encoder_input_dim,
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        state_embed_dim=int(model_cfg.get("state_embed_dim", 64)),
        reward_embed_dim=int(model_cfg.get("reward_embed_dim", 64)),
        num_reward_embeddings=int(model_cfg.get("num_reward_embeddings", 32)),
        num_layers=int(model_cfg.get("num_encoder_blocks", 4)),
        num_heads=int(model_cfg.get("encoder_attention_heads", 4)),
        mlp_dim=int((model_cfg.get("encoder_layers") or [256])[0]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        activation=str(model_cfg.get("activation", "relu")),
            log_std_min=float(model_cfg.get("log_std_min", -5.0)),
            log_std_max=float(model_cfg.get("log_std_max", 2.0)),
        )
    encoder = encoder.to(device)
    checkpoint = first_not_none(args.checkpoint, getattr(args, "encoder_checkpoint", None))
    if checkpoint:
        _load_module_state(encoder, checkpoint, keys=("encoder", "encoder_state_dict"))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def _infer_obs_dim(dataset: Any) -> Optional[int]:
    for name in ("observation_space", "agent_observation_space", "encoder_observation_space"):
        space = getattr(dataset, name, None)
        try:
            if space is not None and hasattr(space, "shape"):
                return int(space.shape[0])
        except Exception:  # pragma: no cover
            continue
    return None


def build_policy(domain: str, dataset: Any, encoder: Any, cfg: Dict[str, Any],
                 args: argparse.Namespace, device: str) -> Any:
    """Build the ``z``-conditioned IQL policy and load its checkpoint if given.

    The RL networks are *not* conditioned on the observation embedding of the encoder;
    per the addendum "the latent embedding is simply concatenated to the observation
    state that is fed into the RL components".
    """
    from fre.models.rl_networks import make_rl_networks

    model_cfg = cfg.get("model", {}) or {}
    policy_cfg = cfg.get("policy_training", {}) or {}
    obs_dim = int(first_not_none(getattr(dataset, "obs_dim", None), _infer_obs_dim(dataset)))
    act_dim = int(first_not_none(getattr(dataset, "act_dim", None), _infer_act_dim(dataset)))

    networks = call_flexibly(
        make_rl_networks,
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        hidden_dims=tuple(model_cfg.get("rl_layers", (512, 512, 512))),
        num_qs=int(model_cfg.get("num_qs", 2)),
        encoder=encoder,
        freeze_encoder=True,
        discount=float(policy_cfg.get("discount", 0.88)),
        tau=float(policy_cfg.get("tau", 0.001)),
        expectile=float(policy_cfg.get("expectile", 0.8)),
        awr_temperature=float(policy_cfg.get("awr_temperature", 3.0)),
        activation=str(model_cfg.get("activation", "relu")),
    )
    networks = networks.to(device) if hasattr(networks, "to") else networks
    if args.policy:
        _load_module_state(
            networks,
            args.policy,
            keys=("networks", "rl_networks", "policy_state_dict", "state_dict"),
        )
    if hasattr(networks, "set_eval_mode"):
        networks.set_eval_mode()
    else:  # pragma: no cover - defensive
        for module in getattr(networks, "modules", lambda: [])():
            if hasattr(module, "eval"):
                module.eval()
    return networks.policy if hasattr(networks, "policy") else networks


def _infer_act_dim(dataset: Any) -> Optional[int]:
    space = getattr(dataset, "observation_space", None)
    action_space = getattr(dataset, "action_space", None)
    if action_space is not None and hasattr(action_space, "shape"):
        return int(action_space.shape[0])
    actions = getattr(dataset, "actions", None)
    if actions is not None and hasattr(actions, "shape") and actions.ndim == 2:
        return int(actions.shape[1])
    return None


def _load_module_state(module: Any, checkpoint: str, keys: Sequence[str] = ()) -> Any:
    """Load a checkpoint that may be a bare ``state_dict`` or a training bundle."""
    _torch = torch()
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = _torch.load(checkpoint, map_location="cpu")
    state = payload
    if isinstance(payload, dict):
        for key in keys:
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
        else:
            if "state_dict" in payload and isinstance(payload["state_dict"], dict):
                state = payload["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {checkpoint} does not contain a state dict")
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[evaluate] loaded {os.path.basename(checkpoint)} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})",
            file=sys.stderr,
        )
    return payload


# ======================================================================================
# Task suites
# ======================================================================================
def build_suites(domain: str, cfg: Dict[str, Any], dataset: Any,
                 suite_names: Sequence[str]) -> Dict[str, Any]:
    """Build the requested zero-shot task suites for a domain.

    Source: Addendum "Ant Maze evaluation tasks" / "ExORL evaluation tasks" and
    Appendix C.1-C.3.  Goal coordinates (AntMaze (28,0) etc.), velocity thresholds
    (cheetah 10/1, walker 0.1/1/4/8, linearly decaying to 0 and 0 when moving against the
    target) and the 0.1 ExORL goal distance all come from the paper and are supplied by
    the ``fre.envs.*_tasks`` modules, not re-derived here.
    """
    eval_cfg = cfg.get("evaluation", {}) or {}
    task_kwargs = eval_cfg.get("task_kwargs", {}) or {}
    episode_length = int(
        first_not_none(
            eval_cfg.get("eval_episode_length"),
            DOMAIN_DEFAULTS.get(domain, {}).get("episode_length"),
        )
    )
    suites: Dict[str, Any] = {}
    domain = domain.lower()

    if domain == "antmaze":
        from fre.envs.antmaze_tasks import make_antmaze_task_suite

        common = {
            "goal_locations": task_kwargs.get("goal_locations"),
            "directions": task_kwargs.get("directions"),
            "velocity_dims": tuple(task_kwargs.get("velocity_dims", (15, 16))),
            "simplex_seeds": tuple(task_kwargs.get("simplex_seeds", (1, 2, 3, 4, 5))),
            "simplex_frequency": task_kwargs.get("simplex_frequency"),
            "simplex_height_bonus": task_kwargs.get("simplex_height_bonus"),
            "simplex_velocity_bonus": task_kwargs.get("simplex_velocity_bonus"),
            "simplex_baseline": task_kwargs.get("simplex_baseline"),
            "path_width": task_kwargs.get("path_width"),
            "threshold": float(
                first_not_none(
                    eval_cfg.get("goal_threshold"),
                    task_kwargs.get("goal_threshold"),
                    2.0,  # Source: Appendix C.1 goal reached within distance 2
                )
            ),
            "eval_episode_length": episode_length,
        }
        for name in suite_names:
            try:
                suites[name] = call_flexibly(
                    make_antmaze_task_suite, group=name, **common
                )
            except Exception as exc:  # pragma: no cover - suite-specific failure
                print(f"[evaluate] skipping AntMaze suite {name!r}: {exc}", file=sys.stderr)

    elif domain == "exorl":
        from fre.envs.exorl_tasks import make_exorl_task_suite

        goal_states = getattr(dataset, "goal_states", None)
        state_mean = getattr(dataset, "state_mean", None)
        state_std = getattr(dataset, "state_std", None)
        common = {
            "goal_states": goal_states,
            "num_goals": int(eval_cfg.get("num_goals", 5)),
            "threshold": float(eval_cfg.get("goal_distance_threshold", 0.1)),
            "state_mean": state_mean,
            "state_std": state_std,
            "agent_obs_dim": getattr(dataset, "obs_dim", None),
            "eval_episode_length": episode_length,
            "cheetah_thresholds": tuple(task_kwargs.get("cheetah_thresholds", (10.0, 1.0))),
            "walker_thresholds": tuple(
                task_kwargs.get("walker_thresholds", (0.1, 1.0, 4.0, 8.0))
            ),
        }
        default_domain = first_not_none(getattr(dataset, "domain", None), "walker")
        for name in suite_names:
            try:
                suites[name] = call_flexibly(
                    make_exorl_task_suite, group=name, domain=default_domain, **common
                )
            except Exception as exc:  # pragma: no cover
                print(f"[evaluate] skipping ExORL suite {name!r}: {exc}", file=sys.stderr)

    elif domain == "kitchen":
        from fre.envs.kitchen_tasks import make_kitchen_task_suite

        common = {
            "threshold": task_kwargs.get("threshold"),
            "eval_episode_length": episode_length,
        }
        for name in suite_names:
            try:
                suites[name] = call_flexibly(make_kitchen_task_suite, group=name, **common)
            except Exception as exc:  # pragma: no cover
                print(f"[evaluate] skipping Kitchen suite {name!r}: {exc}", file=sys.stderr)

    else:  # pragma: no cover
        raise ValueError(f"unknown domain '{domain}'")

    if not suites:
        raise RuntimeError(
            f"no task suites could be constructed for domain '{domain}' "
            f"(requested: {list(suite_names)})"
        )
    return suites


# ======================================================================================
# Environment construction (optional: the harness works with injectable rollout fns)
# ======================================================================================
def make_env_factory(domain: str, cfg: Dict[str, Any], args: argparse.Namespace) -> Optional[Callable[..., Any]]:
    """Return ``env_factory(**kwargs) -> gym env`` for the domain, or ``None``.

    Real simulators are optional: when gym / D4RL / dm_control are unavailable the
    evaluator falls back to whatever ``rollout_fn`` the caller injects (useful for unit
    tests and for offline scoring backends).
    """
    domain = domain.lower()
    if args.no_env:
        return None

    if domain in ("antmaze", "kitchen"):
        env_id = first_not_none(
            cfg_get(cfg, "dataset.env_id"),
            cfg_get(cfg, "dataset.name"),
            DOMAIN_DEFAULTS.get(domain, {}).get("env_id"),
        )

        def _factory(env_id: str = str(env_id), **_kwargs: Any) -> Any:
            import gym  # local import: optional dependency

            try:  # D4RL registers extra env ids (also needed for kitchen-*-v0)
                import d4rl  # noqa: F401
            except Exception:  # pragma: no cover - non-D4RL envs still work
                pass
            env = gym.make(env_id)
            return env

        return _factory

    if domain == "exorl":
        domain_name = str(first_not_none(args.dataset, cfg_get(cfg, "dataset.domain"), "walker"))
        task = str(first_not_none(args.task, cfg_get(cfg, "dataset.task"), "run"))

        def _factory(domain_name: str = domain_name, task: str = task, **_kwargs: Any) -> Any:
            from dm_control import suite  # optional dependency

            max_steps = int(_kwargs.get("max_steps", DOMAIN_DEFAULTS["exorl"]["episode_length"]))
            return suite.load(domain_name=domain_name, task_name=task, task_kwargs={"random": 0},
                              environment_kwargs={"flat_observation": True}) if False else _DmControlEnv(
                domain_name, task, max_steps
            )

        return _factory

    return None


class _DmControlEnv:
    """Thin adapter giving a dm_control task a gym-like ``step``/``reset`` interface.

    Only used when the ExORL simulators are installed; the zero-shot harness consumes
    ``(observation, reward, done, info)`` tuples from ``env.step``.
    """

    def __init__(self, domain_name: str, task_name: str, max_steps: int = 1000,
                 seed: int = 0):
        from dm_control import suite  # pragma: no cover - optional dependency

        self._env = suite.load(domain_name=domain_name, task_name=task_name,
                               task_kwargs={"random": int(seed)})
        self._max_steps = int(max_steps)
        self._step = 0
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **_kwargs: Any) -> np.ndarray:
        timestep = self._env.reset()
        self._step = 0
        self._last_obs = self._flatten(timestep)
        return self._last_obs

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        timestep = self._env.step(np.asarray(action, dtype=np.float64))
        self._step += 1
        obs = self._flatten(timestep)
        self._last_obs = obs
        reward = float(timestep.reward or 0.0)
        done = bool(timestep.last()) or self._step >= self._max_steps
        return obs, reward, done, {}

    @staticmethod
    def _flatten(timestep: Any) -> np.ndarray:
        parts = []
        for value in timestep.observation.values():
            parts.append(np.atleast_1d(np.asarray(value, dtype=np.float32)).ravel())
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    @property
    def action_space(self) -> Any:  # pragma: no cover - trivial accessor
        return self._env.action_spec()


# ======================================================================================
# Evaluation drivers
# ======================================================================================
def run_fre_evaluation(
    encoder: Any,
    policy: Any,
    dataset: Any,
    suites: Dict[str, Any],
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    device: str,
    decoder: Any = None,
) -> Tuple[List[Any], Dict[str, Dict[str, float]], str]:
    """Evaluate the FRE agent zero-shot on every supplied suite / seed."""
    from fre.eval.zero_shot_eval import aggregate_seeds, format_seed_summary
    from fre.eval.zero_shot_eval import ZeroShotEvaluator

    eval_cfg = cfg.get("evaluation", {}) or {}
    context_samples = int(
        first_not_none(args.context_samples, eval_cfg.get("context_samples"), FRE_CONTEXT_SAMPLES)
    )
    num_episodes = int(
        first_not_none(args.num_episodes, eval_cfg.get("num_episodes"), NUM_EVAL_EPISODES)
    )
    use_encoder_inputs = _encoder_inputs_flag(dataset, cfg, args)
    env_factory = make_env_factory(args.domain, cfg, args)

    evaluator = call_flexibly(
        ZeroShotEvaluator,
        encoder=encoder,
        policy=policy,
        replay_buffer=getattr(dataset, "replay_buffer", None),
        decoder=decoder,
        device=device,
        num_context_samples=context_samples,
        num_episodes=num_episodes,
        seeds=tuple(args.seeds),
        deterministic_policy=bool(eval_cfg.get("deterministic_policy", True)),
        use_encoder_inputs=use_encoder_inputs,
        clip_scores=bool(eval_cfg.get("clip_scores", False)),
        sample_latent=bool(eval_cfg.get("sample_latent", False)),
        name="FRE",
    )

    reports: List[Any] = []
    for seed in args.seeds:
        for name, suite in suites.items():
            result = _evaluate_suite_flexibly(
                evaluator, suite, seed=seed, env_factory=env_factory, args=args, dataset=dataset
            )
            result = _retag(result, suite_name=name, seed=seed)
            reports.append(result)
            print(f"[evaluate] FRE seed={seed} suite={name}: {_score_of(result):.1f}")

    summary = call_flexibly(aggregate_seeds, reports)
    return reports, summary, call_flexibly(format_seed_summary, summary)


def _encoder_inputs_flag(dataset: Any, cfg: Dict[str, Any], args: argparse.Namespace) -> Optional[bool]:
    """Whether encoder inputs (e.g. ExORL physics) must be used.

    Source: Appendix C.2 - the auxiliary physics information is used by the encoder
    only; encoder-input sampling therefore has to be requested explicitly for ExORL.
    """
    if args.use_encoder_inputs is not None:
        return bool(args.use_encoder_inputs)
    configured = cfg_get(cfg, "encoder_training.use_encoder_inputs")
    if configured is not None:
        return bool(configured)
    if cfg_get(cfg, "dataset.physics.enabled") and cfg_get(cfg, "dataset.physics.encoder_only", True):
        return True
    return None


def _evaluate_suite_flexibly(evaluator: Any, suite: Any, seed: int,
                             env_factory: Optional[Callable[..., Any]],
                             args: argparse.Namespace, dataset: Any) -> Any:
    """Call whichever ``evaluate*`` entry point the harness exposes."""
    common = {
        "seed": seed,
        "env_factory": env_factory,
        "num_episodes": args.num_episodes,
        "deterministic": True,
        "record_observations": bool(args.record_observations),
    }
    for method_name in ("evaluate_suite", "run_suite", "evaluate"):
        method = getattr(evaluator, method_name, None)
        if method is None:
            continue
        try:
            return call_flexibly(method, suite, **common)
        except TypeError:
            continue
    raise AttributeError(
        "ZeroShotEvaluator exposes none of evaluate_suite/run_suite/evaluate"
    )


def _retag(result: Any, suite_name: str, seed: int) -> Any:
    """Make sure the report carries the suite name/seed used for aggregation."""
    for attr, value in (("suite_name", suite_name), ("seed", seed)):
        try:
            current = getattr(result, attr, None)
            if current in (None, "", "suite"):
                setattr(result, attr, value)
        except Exception:  # pragma: no cover - slots/frozen dataclasses
            pass
    return result


def _score_of(result: Any) -> float:
    for attr in ("mean_score", "score"):
        value = getattr(result, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float("nan")


def build_baseline_agent(method: str, dataset: Any, cfg: Dict[str, Any],
                         args: argparse.Namespace, device: str) -> Any:
    """Construct a GC-BC / GC-IQL / OPAL-10 baseline agent."""
    key = method.upper()
    baselines_cfg = cfg.get("baselines", {}) or {}
    replay_buffer = getattr(dataset, "replay_buffer", None)
    obs_dim = int(first_not_none(getattr(dataset, "obs_dim", None), _infer_obs_dim(dataset)))
    act_dim = int(first_not_none(getattr(dataset, "act_dim", None), _infer_act_dim(dataset)))
    seed = int(args.seed)

    if key == "GC-BC":
        from fre.eval.baselines.gc_bc import make_gc_bc_agent

        agent = call_flexibly(
            make_gc_bc_agent,
            replay_buffer=replay_buffer,
            obs_dim=obs_dim,
            act_dim=act_dim,
            goal_dim=obs_dim,
            device=device,
            seed=seed,
            **{k: v for k, v in (baselines_cfg.get("gc_bc") or {}).items()},
        )
    elif key == "GC-IQL":
        from fre.eval.baselines.gc_iql import make_gc_iql_agent

        agent = call_flexibly(
            make_gc_iql_agent,
            replay_buffer=replay_buffer,
            obs_dim=obs_dim,
            act_dim=act_dim,
            goal_dim=obs_dim,
            device=device,
            seed=seed,
            **{k: v for k, v in (baselines_cfg.get("gc_iql") or {}).items()},
        )
    elif key in ("OPAL-10", "OPAL"):
        from fre.eval.baselines.opal import make_opal_agent

        agent = call_flexibly(
            make_opal_agent,
            replay_buffer=replay_buffer,
            obs_dim=obs_dim,
            act_dim=act_dim,
            latent_dim=int(cfg_get(cfg, "model.latent_dim", 128)),
            device=device,
            seed=seed,
            **{k: v for k, v in (baselines_cfg.get("opal") or {}).items()},
        )
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown baseline method '{method}'")

    if args.policy:
        # Baselines are evaluated from their own trained agent checkpoints.
        try:
            agent.load(args.policy)
        except Exception as exc:  # pragma: no cover
            print(f"[evaluate] could not load baseline checkpoint: {exc}", file=sys.stderr)
    return agent


def run_baseline_evaluation(agent: Any, suites: Dict[str, Any], args: argparse.Namespace,
                            cfg: Dict[str, Any], dataset: Any) -> Tuple[List[Any], Any]:
    """Evaluate a goal-conditioned / skill-based baseline with the shared driver."""
    from fre.eval.baselines import evaluate_baseline, format_baseline_summary

    eval_cfg = cfg.get("evaluation", {}) or {}
    env_factory = make_env_factory(args.domain, cfg, args)
    context_samples = int(
        first_not_none(args.context_samples, eval_cfg.get("context_samples"),
                       FRE_CONTEXT_SAMPLES)
    )
    reports: List[Any] = []
    for seed in args.seeds:
        for name, suite in suites.items():
            result = call_flexibly(
                evaluate_baseline,
                agent,
                [suite],
                seeds=(seed,),
                env_factory=env_factory,
                replay_buffer=getattr(dataset, "replay_buffer", None),
                num_episodes=int(first_not_none(args.num_episodes, eval_cfg.get("num_episodes"),
                                                NUM_EVAL_EPISODES)),
                deterministic=True,
                best_of_skills=True,  # OPAL-10 privileged "best rollout" protocol
                num_context_samples=context_samples,
                method=agent.name,
            )
            got = result if isinstance(result, list) else [result]
            for item in got:
                item = _retag(item, suite_name=name, seed=seed)
                reports.append(item)
                print(f"[evaluate] {agent.name} seed={seed} suite={name}: {_score_of(item):.1f}")

    try:
        return reports, format_baseline_summary(_aggregate_seeds(reports))
    except Exception:  # pragma: no cover - summary is cosmetic
        return reports, {}


def _aggregate_seeds(reports: List[Any]) -> Dict[str, Dict[str, float]]:
    from fre.eval.baselines import aggregate_baseline_seeds

    return aggregate_baseline_seeds(reports)


# ======================================================================================
# Reporting
# ======================================================================================
def save_results(args: argparse.Namespace, summary: Any, reports: List[Any],
                 text: Optional[str] = None) -> Optional[str]:
    """Persist the aggregated summary + per-report details as JSON."""
    if not args.save:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "method": args.method,
        "domain": args.domain,
        "dataset": args.dataset,
        "seeds": list(args.seeds),
        "num_episodes": int(args.num_episodes),
        "context_samples": int(args.context_samples) if args.context_samples else None,
        "suites": list(args.suites or []),
        "summary": _jsonify(summary),
        "text": text,
        "reports": [_jsonify(_to_dict_if_possible(r)) for r in reports],
        "protocol": {
            "num_episodes": NUM_EVAL_EPISODES,
            "num_seeds": NUM_TRAINING_SEEDS,
            "fre_context_samples": FRE_CONTEXT_SAMPLES,
            "fb_sf_context_samples": FB_SF_CONTEXT_SAMPLES,
            "normalized_return_range": [NORMALIZED_RETURN_MIN, NORMALIZED_RETURN_MAX],
            "source": "Section 5.2 / Table 1 caption",
        },
    }
    with open(args.save, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"[evaluate] wrote {args.save}")
    return args.save


def _to_dict_if_possible(obj: Any) -> Any:
    for name in ("to_dict",):
        method = getattr(obj, name, None)
        if callable(method):
            try:
                return method()
            except Exception:  # pragma: no cover
                pass
    return obj


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def print_summary(method: str, summary: Any, text: Optional[str]) -> None:
    print("=" * 70)
    print(f"{method} zero-shot evaluation (20 episodes/task, 5 seeds, 0-100 scale)")
    print("=" * 70)
    if text:
        print(text)
    elif isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, dict):
                mean = value.get("mean")
                std = value.get("std")
                print(f"{key:<32} {_fmt(mean)} +- {_fmt(std)}")
            else:
                print(f"{key:<32} {_fmt(value)}")
    print("=" * 70)


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


# ======================================================================================
# CLI
# ======================================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Zero-shot evaluation of FRE (and the GC-BC / GC-IQL / OPAL-10 baselines) on "
            "AntMaze / ExORL / Kitchen, following Section 5.2 of "
            "'Zero-Shot Reinforcement Learning via Functional Reward Encodings'."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config (configs/fre_{antmaze,exorl,kitchen}.yaml)")
    parser.add_argument("--domain", type=str, default="antmaze",
                        choices=["antmaze", "exorl", "kitchen"])
    parser.add_argument("--method", type=str, default="FRE", choices=list(METHODS))
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset path (AntMaze/Kitchen) or ExORL domain "
                             "(walker|cheetah)")
    parser.add_argument("--dataset-dir", dest="dataset_dir", type=str, default=None,
                        help="directory holding ExORL HDF5 dumps")
    parser.add_argument("--task", type=str, default=None,
                        help="ExORL task name (run|walk|run-backwards|walk-backwards)")
    parser.add_argument("--checkpoint", "--encoder-checkpoint", dest="checkpoint",
                        type=str, default=None,
                        help="FRE encoder checkpoint (phase-1 output)")
    parser.add_argument("--policy", type=str, default=None,
                        help="policy/agent checkpoint (phase-2 output) or baseline agent")
    parser.add_argument("--suites", nargs="*", default=None,
                        help="task suite groups (default: all groups of the domain)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="training seeds to evaluate (paper uses 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed used for dataset/goal construction")
    parser.add_argument("--num-episodes", dest="num_episodes", type=int,
                        default=NUM_EVAL_EPISODES,
                        help="evaluation episodes per task (paper: 20)")
    parser.add_argument("--context-samples", dest="context_samples", type=int, default=None,
                        help="reward-annotated states encoded into z (FRE: 32, FB/SF: 5120)")
    parser.add_argument("--use-encoder-inputs", dest="use_encoder_inputs",
                        action="store_true", default=None,
                        help="feed physics-augmented observations to the encoder (ExORL)")
    parser.add_argument("--no-encoder-inputs", dest="use_encoder_inputs",
                        action="store_false", default=None,
                        help="feed the plain observation space to the encoder")
    parser.add_argument("--record-observations", dest="record_observations",
                        action="store_true", default=False,
                        help="store rollout observations in the report")
    parser.add_argument("--no-env", dest="no_env", action="store_true", default=False,
                        help="do not build simulators (offline scoring / smoke test)")
    parser.add_argument("--save", type=str, default=None, help="write results JSON here")
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|cuda:0")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                        help="print the resolved evaluation plan and exit")
    return parser


def resolve_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """Merge CLI arguments with config values and paper defaults."""
    eval_cfg = cfg.get("evaluation", {}) or {}
    trainer_cfg = cfg.get("training", {}) or {}

    args.seeds = ensure_sequence(
        first_not_none(args.seeds, cfg.get("seeds"), trainer_cfg.get("seeds")),
        default=tuple(range(NUM_TRAINING_SEEDS)),
    )
    args.suites = resolve_suites(
        args.domain, args.suites, config=eval_cfg.get("suites")
    )
    args.num_episodes = int(
        first_not_none(args.num_episodes, eval_cfg.get("num_episodes"), NUM_EVAL_EPISODES)
    )
    args.context_samples = int(
        first_not_none(args.context_samples, eval_cfg.get("context_samples"),
                       FRE_CONTEXT_SAMPLES)
    )
    args.device = resolve_device(first_not_none(args.device, cfg.get("device")))
    if args.save is None:
        args.save = eval_cfg.get("save_path")
    if args.dataset_dir is None:
        args.dataset_dir = os.environ.get("FRE_EXORL_DATASET_DIR")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    args = resolve_args(args, cfg)

    if args.dry_run:
        print(json.dumps(
            {
                "domain": args.domain,
                "method": args.method,
                "suites": args.suites,
                "seeds": args.seeds,
                "num_episodes": args.num_episodes,
                "context_samples": args.context_samples,
                "device": args.device,
                "encoder_checkpoint": args.checkpoint,
                "policy_checkpoint": args.policy,
            },
            indent=2,
        ))
        return 0

    method = args.method.upper()
    try:
        dataset = build_dataset(args.domain, cfg, args)
        suites = build_suites(args.domain, cfg, dataset, args.suites)

        if method == "FRE":
            encoder = build_encoder(args.domain, dataset, cfg, args, args.device)
            policy = build_policy(args.domain, dataset, encoder, cfg, args, args.device)
            reports, summary, text = run_fre_evaluation(
                encoder, policy, dataset, suites, args, cfg, args.device
            )
        else:
            agent = build_baseline_agent(method, dataset, cfg, args, args.device)
            reports, summary = run_baseline_evaluation(agent, suites, args, cfg, dataset)
            text = None

        print_summary(method, summary, text)
        save_results(args, summary, reports, text)
        return 0
    except Exception:  # pragma: no cover - surface the full traceback, return non-zero
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
