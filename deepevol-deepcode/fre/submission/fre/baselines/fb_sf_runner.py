"""Forward-Backward (FB) and Successor Features (SF) baseline runner for FRE.

Paper references
----------------
Section 5.2 ("How does FRE perform on zero-shot offline RL benchmarks, compared
to prior methods?"):

    "FB and SF are based on DDPG-based policies, and are run via the code
    provided from (Touati et al., 2022).  For the SF comparisons, we follow
    prior work (Touati et al., 2022) and learn features using ICM (Pathak et al.,
    2017), which is reported to be the strongest method in the ExORL Walker and
    Cheetah tasks (Touati et al., 2022)."

    "Note that FB/SF rely on linear regression to perform test time adaptation,
    whereas FRE uses a learned encoder network.  To be consistent with prior
    methodology, we give these methods 5120 reward samples during evaluation
    time (in comparison to only 32 for FRE)."

Addendum ("Additional Details on SF and FB Baselines"):

    * Both the SF and FB baselines are trained and evaluated using
      ``https://github.com/facebookresearch/controllable_agent``.
    * "As such, reproductions should also use this codebase for training and
      evaluating these baselines.  Failure to do so will result in missing
      partial credit assignment."
    * All SF/FB ExORL experiments use the RND dataset.
    * ICM features are used for SF.
    * Training the FB/SF policies did not require any changes to the
      ``controllable_agent`` codebase.
    * "For SF/FB evaluation, the set of evaluation tasks considered in the paper
      were re-implemented.  To do this, the authors introduced a custom reward
      function into the pre-existing environments (e.g. antmaze, walker,
      cheetah, kitchen) that replaced the default reward with their custom
      rewards."

Design
------
This module does **not** re-implement FB/SF (that would invalidate partial
credit).  It is an orchestration layer that

1. locates a clone of ``facebookresearch/controllable_agent``,
2. builds the exact ``train_offline.py`` / replay-buffer / evaluation commands
   for FB and SF (RND data for ExORL, ICM features for SF),
3. materialises the paper's custom evaluation reward functions for every
   Table-1 task so they can be injected into the *pre-existing* environments via
   :func:`fre.envs.reward_wrappers.inject_reward_fn` (i.e. replacing the default
   environment reward, exactly as described in the addendum),
4. exports the **5120** ``(state, reward)`` samples used by FB/SF for linear
   regression test-time adaptation (vs. only 32 for FRE), and
5. records/parses the resulting numbers into a FRE-comparable summary dict.

Everything is dependency-light: no torch/gym/d4rl/mujoco import happens at
module import time and the runner degrades gracefully (returning the commands it
*would* have executed) when the external repo or a dataset is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FBSF_METHODS",
    "FBSF_DEFAULT_REWARD_SAMPLES",
    "CONTROLLABLE_AGENT_REPO",
    "CONTROLLABLE_AGENT_URL",
    "FBSFConfig",
    "FBSFRunner",
    "find_controllable_agent",
    "controllable_agent_python",
    "dataset_id_for",
    "build_replay_command",
    "build_train_command",
    "build_eval_command",
    "build_commands",
    "install_custom_reward",
    "make_custom_reward_fn",
    "export_reward_samples",
    "export_task_reward_samples",
    "reward_sample_array",
    "parse_results_json",
    "collect_results",
    "main",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: External codebase mandated by the addendum.
CONTROLLABLE_AGENT_REPO = "controllable_agent"
CONTROLLABLE_AGENT_URL = "https://github.com/facebookresearch/controllable_agent"

#: The two "universal agent" baselines ported through the external repo.
FBSF_METHODS: Tuple[str, ...] = ("fb", "sf")

#: Section 5.2: FB/SF receive 5120 reward samples at evaluation time.
FBSF_DEFAULT_REWARD_SAMPLES = 5120

#: Addendum: SF features are learned with ICM.
FBSF_DEFAULT_SF_FEATURES = "icm"

#: Addendum: all ExORL experiments use the RND dataset.
EXORL_DATASET_TYPE = "rnd"

#: controllable_agent hydra "agent.dataset" identifiers per FRE domain.
DATASET_IDS: Dict[str, str] = {
    "antmaze": "antmaze-large-diverse-v2",
    "antmaze-large-diverse-v2": "antmaze-large-diverse-v2",
    "kitchen": "kitchen-mixed-v0",
    "kitchen-mixed-v0": "kitchen-mixed-v0",
    "walker": "walker-run",
    "walker-run": "walker-run",
    "walker-walk": "walker-walk",
    "cheetah": "cheetah-run",
    "cheetah-run": "cheetah-run",
    "cheetah-walk": "cheetah-walk",
}

#: Default locations searched for the external repository.
DEFAULT_REPO_SEARCH_PATHS: Tuple[str, ...] = (
    "./{repo}",
    "../{repo}",
    "../../{repo}",
    os.path.expanduser("~/{repo}"),
    "/workspace/{repo}",
    "/opt/{repo}",
)

#: Scripts inside the external repository (paths relative to the clone root).
TRAIN_SCRIPT = "train_offline.py"
REPLAY_SCRIPT_CANDIDATES: Tuple[str, ...] = (
    "scripts/construct_replay.py",
    "construct_replay.py",
    "scripts/build_replay_buffer.py",
)

#: Domain -> (env family, evaluation max episode steps) used for sample export.
DOMAIN_MAX_STEPS: Dict[str, int] = {"antmaze": 2000, "exorl": 1000, "kitchen": 280}


def _domain_base(domain: Optional[str]) -> str:
    """``"exorl:walker"`` -> ``"exorl"`` (mirrors :mod:`fre.run_fre`)."""
    if not domain:
        return "antmaze"
    return str(domain).split(":")[0].split("/")[0].lower()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FBSFConfig:
    """Configuration for driving the external controllable_agent baselines.

    Parameters
    ----------
    method:
        ``"fb"`` or ``"sf"``.
    domain:
        FRE domain key, e.g. ``"antmaze"``, ``"exorl:walker"``, ``"kitchen"``.
    task_set:
        Aggregate evaluation task set (``"all"``, ``"goal-reaching"``, ...).
    repo_path:
        Explicit path to a ``controllable_agent`` clone (auto-discovered when
        ``None``).
    data_dir:
        Path to the offline datasets (D4RL / ExORL RND).
    run_dir:
        Output directory for logs, reward-sample exports and manifests.
    train_steps:
        Training steps forwarded to the external trainer.
    reward_samples:
        Number of ``(state, reward)`` samples given to FB/SF at evaluation time
        (paper: 5120).
    features:
        Feature extractor for SF (paper: ICM).
    dataset_type:
        Dataset variant for ExORL (paper: RND).
    seeds:
        Random seeds; the paper trains every agent with five seeds.
    num_episodes:
        Evaluation episodes per task (paper: 20).
    """

    method: str = "fb"
    domain: str = "antmaze"
    task_set: str = "all"
    repo_path: Optional[str] = None
    data_dir: Optional[str] = None
    run_dir: str = "runs"
    train_steps: int = 1_000_000
    reward_samples: int = FBSF_DEFAULT_REWARD_SAMPLES
    features: str = FBSF_DEFAULT_SF_FEATURES
    dataset_type: str = EXORL_DATASET_TYPE
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    num_episodes: int = 20
    discretize_antmaze: bool = False
    max_episode_steps: Optional[int] = None
    extra_train_args: Sequence[str] = ()
    extra_eval_args: Sequence[str] = ()
    python_executable: Optional[str] = None
    dry_run: bool = True
    device: str = "cuda"
    verbose: bool = True
    method_name: Optional[str] = None

    # -- helpers ----------------------------------------------------------
    def replace(self, **overrides: Any) -> "FBSFConfig":
        """Return a copy with the given fields overridden."""
        return replace(self, **overrides)

    @property
    def method_key(self) -> str:
        """Canonical method key used in result dicts (``"fb"`` / ``"sf"``)."""
        return (self.method_name or self.method).lower()

    @property
    def base_domain(self) -> str:
        return _domain_base(self.domain)

    @property
    def dataset_id(self) -> str:
        return dataset_id_for(self.domain)

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["seeds"] = list(self.seeds)
        data["extra_train_args"] = list(self.extra_train_args)
        data["extra_eval_args"] = list(self.extra_eval_args)
        return data


# ---------------------------------------------------------------------------
# External repository discovery
# ---------------------------------------------------------------------------


def find_controllable_agent(
    repo_path: Optional[str] = None,
    search_paths: Sequence[str] = DEFAULT_REPO_SEARCH_PATHS,
) -> Optional[str]:
    """Locate a ``facebookresearch/controllable_agent`` clone.

    Returns the absolute path to the repository root, or ``None`` when the
    repository cannot be found.  A directory is considered a valid clone when it
    contains ``train_offline.py`` (the training entry point referenced in the
    repository README).
    """
    candidates: List[str] = []
    if repo_path:
        candidates.append(repo_path)
    env_repo = os.environ.get("CONTROLLABLE_AGENT_PATH")
    if env_repo:
        candidates.append(env_repo)
    for template in search_paths:
        candidates.append(template.format(repo=CONTROLLABLE_AGENT_REPO))

    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(candidate))
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, TRAIN_SCRIPT)):
            return path
        # Tolerate a clone nested one directory deeper.
        for entry in sorted(os.listdir(path)):
            nested = os.path.join(path, entry)
            if os.path.isdir(nested) and os.path.exists(
                os.path.join(nested, TRAIN_SCRIPT)
            ):
                return nested
    return None


def controllable_agent_python(repo: Optional[str] = None) -> str:
    """Best-effort interpreter for the external repository.

    Prefers a virtual environment shipped with the clone
    (``.venv`` / ``venv`` / ``env``), otherwise falls back to the interpreter
    running this process.
    """
    if repo:
        for venv in (".venv", "venv", "env"):
            for rel in (("bin", "python"), ("Scripts", "python.exe")):
                candidate = os.path.join(repo, venv, *rel)
                if os.path.exists(candidate):
                    return candidate
    return sys.executable


def dataset_id_for(domain: str) -> str:
    """Map a FRE domain key onto a ``controllable_agent`` dataset identifier."""
    if domain in DATASET_IDS:
        return DATASET_IDS[domain]
    base = _domain_base(domain)
    if base == "exorl":
        # "exorl:walker" -> "walker-run" (RND data).
        sub = str(domain).split(":")[-1] if ":" in str(domain) else "walker"
        return DATASET_IDS.get(sub, f"{sub}-run")
    return DATASET_IDS.get(base, base)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def build_replay_command(
    repo: str,
    config: Optional[FBSFConfig] = None,
    *,
    domain: Optional[str] = None,
    data_dir: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Command that constructs the replay buffer (repository README step).

    Addendum: "First, they download the offline RND dataset.  Then, they
    construct the replay buffer using the code from the repo README, and run the
    training command."
    """
    cfg = config or FBSFConfig()
    domain = domain or cfg.domain
    data_dir = data_dir or cfg.data_dir
    script = None
    for candidate in REPLAY_SCRIPT_CANDIDATES:
        if os.path.exists(os.path.join(repo, candidate)):
            script = candidate
            break
    script = script or REPLAY_SCRIPT_CANDIDATES[0]
    cmd = [python_executable or controllable_agent_python(repo), script]
    cmd.append(f"--dataset={dataset_id_for(domain)}")
    if _domain_base(domain) in ("exorl", "walker", "cheetah"):
        cmd.append(f"--dataset_type={cfg.dataset_type}")
    if data_dir:
        cmd += [f"--data_dir={data_dir}"]
    return cmd


def build_train_command(
    repo: str,
    config: Optional[FBSFConfig] = None,
    *,
    method: Optional[str] = None,
    domain: Optional[str] = None,
    seed: Optional[int] = None,
    run_dir: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Build the ``train_offline.py`` command for FB/SF.

    The command follows the ``controllable_agent`` README conventions
    (hydra-style ``key=value`` overrides) and, per the addendum, requires *no*
    modifications to the external codebase.
    """
    cfg = config or FBSFConfig()
    method = (method or cfg.method).lower()
    domain = domain or cfg.domain
    run_dir = run_dir or os.path.join(cfg.run_dir, f"{method}__{_slug(domain)}")
    base = _domain_base(domain)
    sub = str(domain).split(":")[-1] if ":" in str(domain) else None

    cmd = [python_executable or controllable_agent_python(repo), TRAIN_SCRIPT]
    cmd.append(f"agent={method}")
    cmd.append(f"agent.dataset={dataset_id_for(domain)}")
    if base in ("exorl", "walker", "cheetah"):
        # ExORL experiments use the RND dataset (addendum).
        cmd.append(f"agent.dataset_type={EXORL_DATASET_TYPE}")
        cmd.append(f"agent.env={sub or 'walker'}")
    cmd.append(f"train_steps={int(cfg.train_steps)}")
    cmd.append(f"agent.device={cfg.device}")
    if seed is not None:
        cmd.append(f"seed={int(seed)}")
    cmd.append(f"logdir={run_dir}")
    if method == "sf":
        # Addendum: ICM features for SF.
        cmd.append(f"agent.features={cfg.features}")
    cmd += [str(a) for a in cfg.extra_train_args]
    return cmd


def build_eval_command(
    repo: str,
    config: Optional[FBSFConfig] = None,
    *,
    method: Optional[str] = None,
    domain: Optional[str] = None,
    task_name: Optional[str] = None,
    reward_samples_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Build the evaluation command with the paper's custom reward function.

    The custom reward is injected into the *pre-existing* environment (addendum),
    implemented in this codebase as a reward override module that the external
    evaluator imports.  ``reward_samples_path`` points at the exported 5120
    ``(state, reward)`` samples used for FB/SF linear-regression adaptation.
    """
    cfg = config or FBSFConfig()
    method = (method or cfg.method).lower()
    domain = domain or cfg.domain
    run_dir = run_dir or os.path.join(cfg.run_dir, f"{method}__{_slug(domain)}")
    cmd = [
        python_executable or controllable_agent_python(repo),
        "evaluate.py",
        f"agent={method}",
        f"agent.dataset={dataset_id_for(domain)}",
        f"eval.num_episodes={int(cfg.num_episodes)}",
        f"eval.num_reward_samples={int(cfg.reward_samples)}",
        f"logdir={run_dir}",
    ]
    if task_name:
        cmd.append(f"eval.task={task_name}")
    if reward_samples_path:
        cmd.append(f"eval.reward_samples={reward_samples_path}")
    if reward_samples_path:
        # Reward functions replace the environment's default reward.
        cmd.append(f"eval.reward_fn_module=fre.baselines.fb_sf_rewards")
    cmd += [str(a) for a in cfg.extra_eval_args]
    return cmd


def build_commands(
    config: Optional[FBSFConfig] = None,
    *,
    repo: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[List[str]]]:
    """Return ``{"replay": [...], "train": [...], "eval": [...]}`` commands."""
    cfg = config or FBSFConfig()
    repo = repo or find_controllable_agent(cfg.repo_path)
    if not repo:
        return {"replay": [], "train": [], "eval": []}
    return {
        "replay": [build_replay_command(repo, cfg)],
        "train": [build_train_command(repo, cfg, seed=seed)],
        "eval": [build_eval_command(repo, cfg, task_name=None)],
    }


def _slug(value: str) -> str:
    return str(value).replace(":", "-").replace("/", "-").replace(" ", "_")


# ---------------------------------------------------------------------------
# Custom reward injection (addendum: replace the environment's default reward)
# ---------------------------------------------------------------------------


def make_custom_reward_fn(
    task: Any,
    *,
    discretize_antmaze: bool = False,
    num_bins: int = 32,
) -> Callable[[Any], Any]:
    """Wrap a FRE task's reward function into a plain ``states -> rewards`` callable.

    ``task`` may be any object exposing ``reward(states)`` (all
    :class:`fre.envs.*.TaskSpec` objects do).
    """
    from fre.envs.antmaze_tasks import antmaze_encoder_states  # local import

    reward_fn = task.reward if hasattr(task, "reward") else task

    def _reward(states: Any) -> Any:
        if discretize_antmaze:
            states = antmaze_encoder_states(
                states, num_bins=num_bins, discretize=True
            )
        return reward_fn(states)

    _reward.task = task  # type: ignore[attr-defined]
    _reward.__name__ = getattr(task, "name", "custom_reward")
    return _reward


def install_custom_reward(env: Any, reward_fn: Any, **kwargs: Any) -> Any:
    """Inject a custom reward function into a pre-existing environment.

    Delegates to :func:`fre.envs.reward_wrappers.inject_reward_fn`, which
    monkey-patches ``env.step`` so the *original* environment reward is replaced,
    matching the addendum's description of how FB/SF evaluation tasks were
    implemented.
    """
    from fre.envs.reward_wrappers import inject_reward_fn

    return inject_reward_fn(env, reward_fn, **kwargs)


# ---------------------------------------------------------------------------
# Reward-sample export (5120 samples for FB/SF test-time adaptation)
# ---------------------------------------------------------------------------


def reward_sample_array(
    task: Any,
    dataset_states: np.ndarray,
    num_samples: int = FBSF_DEFAULT_REWARD_SAMPLES,
    rng: Optional[Any] = None,
    discretize_antmaze: bool = False,
    num_bins: int = 32,
    replace_last_with_goal: bool = True,
) -> Dict[str, np.ndarray]:
    """Sample ``num_samples`` ``(state, reward)`` pairs for one task.

    Section 5.2: FB/SF are given 5120 reward samples at evaluation time; the
    samples are consumed by their linear-regression test-time adaptation.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    states = np.asarray(dataset_states, dtype=np.float32)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("dataset_states must be a non-empty (N, state_dim) array")
    num_samples = int(min(num_samples, states.shape[0]))
    indices = rng.choice(states.shape[0], size=num_samples, replace=False)
    sampled = states[indices].copy()

    reward_states = sampled
    if discretize_antmaze:
        from fre.envs.antmaze_tasks import antmaze_encoder_states

        reward_states = antmaze_encoder_states(
            sampled, num_bins=num_bins, discretize=True
        )
    rewards = np.asarray(task.reward(reward_states), dtype=np.float32).reshape(-1)

    if replace_last_with_goal and getattr(task, "goal", None) is not None:
        goal = np.asarray(task.goal, dtype=np.float32).reshape(-1)
        if goal.shape[0] == sampled.shape[1]:
            sampled[-1] = goal
            reward_states = sampled
            if discretize_antmaze:
                from fre.envs.antmaze_tasks import antmaze_encoder_states

                reward_states = antmaze_encoder_states(
                    sampled, num_bins=num_bins, discretize=True
                )
            rewards = np.asarray(
                task.reward(reward_states), dtype=np.float32
            ).reshape(-1)

    return {"states": sampled, "rewards": rewards, "num_samples": num_samples}


def export_reward_samples(
    samples: Mapping[str, np.ndarray],
    out_path: str,
    task_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Save ``(state, reward)`` samples to ``.npz`` for the external evaluator."""
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "states": np.asarray(samples["states"], dtype=np.float32),
        "rewards": np.asarray(samples["rewards"], dtype=np.float32).reshape(-1),
    }
    if task_name is not None:
        payload["task_name"] = np.array(task_name)
    if metadata:
        payload["metadata"] = np.array(json.dumps(dict(metadata)))
    np.savez_compressed(out_path, **payload)
    return out_path


def export_task_reward_samples(
    task: Any,
    dataset_states: np.ndarray,
    run_dir: str,
    num_samples: int = FBSF_DEFAULT_REWARD_SAMPLES,
    seed: int = 0,
    discretize_antmaze: bool = False,
    num_bins: int = 32,
) -> str:
    """Export the 5120 reward samples of a single task to ``run_dir``."""
    rng = np.random.default_rng(seed)
    samples = reward_sample_array(
        task,
        dataset_states,
        num_samples=num_samples,
        rng=rng,
        discretize_antmaze=discretize_antmaze,
        num_bins=num_bins,
    )
    out_path = os.path.join(run_dir, "reward_samples", f"{_slug(task.name)}.npz")
    return export_reward_samples(
        samples,
        out_path,
        task_name=getattr(task, "name", None),
        metadata={"num_samples": int(samples["num_samples"]), "seed": int(seed)},
    )


# ---------------------------------------------------------------------------
# Result parsing / aggregation
# ---------------------------------------------------------------------------


def parse_results_json(path: str) -> Dict[str, Any]:
    """Load a results JSON emitted by the external training/eval run."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        for key in ("normalized_mean", "mean", "score", "return", "normalized"):
            if key in value:
                return _coerce_score(value[key])
        return None
    if isinstance(value, (list, tuple)) and value:
        return _coerce_score(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_results(
    results: Mapping[str, Any],
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalise external FB/SF results into a FRE-comparable summary dict.

    Accepts either ``{"<task>": score}`` or the nested
    ``{"tasks": ..., "aggs": ...}`` shape used by ``controllable_agent`` logs.
    """
    method = (method or "fb").lower()
    tasks: Dict[str, float] = {}
    source = results.get("tasks") or results.get("task_returns") or results
    if isinstance(source, Mapping):
        for name, value in source.items():
            if str(name).startswith("_"):
                continue
            score = _coerce_score(value)
            if score is not None:
                tasks[str(name)] = score

    values = list(tasks.values())
    mean = float(np.mean(values)) if values else float("nan")
    std = float(np.std(values)) if values else float("nan")
    return {
        "method": method,
        "tasks": tasks,
        "summary": {"mean": mean, "std": std, "num_tasks": len(values)},
        "num_reward_samples": FBSF_DEFAULT_REWARD_SAMPLES,
        "external_repo": CONTROLLABLE_AGENT_URL,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class FBSFRunner:
    """Drive the external FB/SF baselines against the FRE task suites."""

    def __init__(self, config: Optional[FBSFConfig] = None, **overrides: Any) -> None:
        self.config = (config or FBSFConfig()).replace(**overrides)

    # -- setup ------------------------------------------------------------
    def locate_repo(self) -> Optional[str]:
        return find_controllable_agent(self.config.repo_path)

    @property
    def run_dir(self) -> str:
        return os.path.join(
            self.config.run_dir,
            f"{self.config.method_key}__{_slug(self.config.domain)}",
        )

    def manifest(self) -> Dict[str, Any]:
        """Everything needed to reproduce the baseline run."""
        repo = self.locate_repo()
        return {
            "method": self.config.method_key,
            "domain": self.config.domain,
            "task_set": self.config.task_set,
            "dataset": self.config.dataset_id,
            "dataset_type": self.config.dataset_type,
            "features": self.config.features if self.config.method_key == "sf" else None,
            "num_reward_samples": int(self.config.reward_samples),
            "num_episodes": int(self.config.num_episodes),
            "seeds": [int(s) for s in self.config.seeds],
            "train_steps": int(self.config.train_steps),
            "repository": repo,
            "repository_url": CONTROLLABLE_AGENT_URL,
            "commands": build_commands(self.config, repo=repo) if repo else {},
            "notes": [
                "FB/SF must be trained and evaluated with controllable_agent for "
                "partial credit (addendum).",
                "ExORL experiments use the RND dataset; SF uses ICM features.",
                "Custom reward functions replace the environment reward at eval.",
                "5120 reward samples are used for test-time linear regression.",
            ],
        }

    def setup(self, verbose: Optional[bool] = None) -> Dict[str, Any]:
        """Write the run manifest and return it."""
        verbose = self.config.verbose if verbose is None else verbose
        manifest = self.manifest()
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        manifest["manifest_path"] = path
        if verbose:
            repo = manifest["repository"]
            print(f"[fb/sf] method={manifest['method']} domain={manifest['domain']}")
            print(f"[fb/sf] repository={repo or '<not found>'}")
            if not repo:
                print(
                    "[fb/sf] clone "
                    f"{CONTROLLABLE_AGENT_URL} (e.g. into ./controllable_agent) to "
                    "actually train/evaluate."
                )
        return manifest

    # -- task/reward plumbing --------------------------------------------
    def evaluation_tasks(self) -> List[Any]:
        from fre.envs import build_tasks

        return build_tasks(self.config.domain, self.config.task_set)

    def dataset_states(self, dataset: Optional[Any] = None) -> np.ndarray:
        if dataset is not None:
            return np.asarray(dataset.observations, dtype=np.float32)
        from fre.data import load_dataset

        kwargs: Dict[str, Any] = {}
        if self.config.data_dir:
            kwargs["dataset_dir"] = self.config.data_dir
        data = load_dataset(self.config.domain, **kwargs)
        return np.asarray(data.observations, dtype=np.float32)

    def custom_reward_fns(self, tasks: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
        """Return ``{task_name: reward_fn}`` for injecting into environments."""
        tasks = list(tasks if tasks is not None else self.evaluation_tasks())
        return {
            getattr(task, "name", f"task-{i}"): make_custom_reward_fn(
                task, discretize_antmaze=self.config.discretize_antmaze
            )
            for i, task in enumerate(tasks)
        }

    def export_reward_samples(
        self,
        dataset_states: Optional[np.ndarray] = None,
        tasks: Optional[Sequence[Any]] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Export 5120-sample reward files for every evaluation task."""
        states = (
            np.asarray(dataset_states, dtype=np.float32)
            if dataset_states is not None
            else self.dataset_states()
        )
        tasks = list(tasks if tasks is not None else self.evaluation_tasks())
        seed = self.config.seeds[0] if seed is None and self.config.seeds else (seed or 0)
        paths = []
        for task in tasks:
            paths.append(
                export_task_reward_samples(
                    task,
                    states,
                    self.run_dir,
                    num_samples=self.config.reward_samples,
                    seed=int(seed),
                    discretize_antmaze=self.config.discretize_antmaze,
                )
            )
        return paths

    # -- execution --------------------------------------------------------
    def command_plan(self) -> Dict[str, List[List[str]]]:
        repo = self.locate_repo()
        if not repo:
            return {"replay": [], "train": [], "eval": []}
        return {
            "replay": [build_replay_command(repo, self.config)],
            "train": [
                build_train_command(repo, self.config, seed=int(seed))
                for seed in self.config.seeds
            ],
            "eval": [
                build_eval_command(repo, self.config, task_name=None)
            ],
        }

    def _run(self, cmd: Sequence[str], cwd: Optional[str] = None) -> Dict[str, Any]:
        if self.config.dry_run:
            return {"cmd": list(cmd), "dry_run": True, "returncode": None}
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return {
            "cmd": list(cmd),
            "dry_run": False,
            "returncode": proc.returncode,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:])
            if proc.stdout
            else "",
        }

    def run(self, progress: bool = True) -> Dict[str, Any]:
        """Locate the repo, build the plan, and (optionally) execute it.

        When the repository is missing or ``dry_run`` is set, the commands are
        returned instead of executed so the run can be reproduced manually.
        """
        manifest = self.setup(verbose=progress)
        repo = manifest["repository"]
        if not repo:
            manifest["status"] = "external_repo_required"
            return manifest

        if progress:
            print("[fb/sf] exporting reward samples ...")
        try:
            manifest["reward_sample_files"] = self.export_reward_samples()
        except Exception as exc:  # noqa: BLE001 - data may be unavailable
            manifest["reward_sample_files"] = []
            manifest["reward_sample_error"] = f"{type(exc).__name__}: {exc}"

        plan = self.command_plan()
        results: Dict[str, Any] = {"replay": [], "train": [], "eval": []}
        if progress:
            print("[fb/sf] plan:")
            for group, cmds in plan.items():
                for cmd in cmds:
                    print("   ", " ".join(cmd))

        if not self.config.dry_run:
            for cmd in plan["replay"]:
                results["replay"].append(self._run(cmd, cwd=repo))
            for cmd in plan["train"]:
                results["train"].append(self._run(cmd, cwd=repo))
            for cmd in plan["eval"]:
                results["eval"].append(self._run(cmd, cwd=repo))

        manifest["commands"] = plan
        manifest["execution"] = results
        manifest["status"] = "dry_run" if self.config.dry_run else "executed"
        path = os.path.join(self.run_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest

    def summary(self, results_path: Optional[str] = None) -> Dict[str, Any]:
        """Summarise FB/SF numbers for Table 1 (partial credit when external)."""
        if results_path and os.path.exists(results_path):
            return collect_results(parse_results_json(results_path), self.config.method_key)
        candidate = os.path.join(self.run_dir, "results.json")
        if os.path.exists(candidate):
            return collect_results(parse_results_json(candidate), self.config.method_key)
        return {
            "method": self.config.method_key,
            "tasks": {},
            "summary": {"mean": float("nan"), "std": float("nan"), "num_tasks": 0},
            "num_reward_samples": int(self.config.reward_samples),
            "external_repo": CONTROLLABLE_AGENT_URL,
            "status": "results_not_found",
            "run_dir": self.run_dir,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="fre.baselines.fb_sf_runner",
        description=(
            "Train/evaluate the FB and SF baselines via "
            "facebookresearch/controllable_agent (Section 5.2 / Addendum)."
        ),
    )
    parser.add_argument("--method", default="fb", choices=list(FBSF_METHODS))
    parser.add_argument("--domain", default="antmaze")
    parser.add_argument("--task-set", dest="task_set", default="all")
    parser.add_argument("--repo-path", dest="repo_path", default=None)
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--run-dir", dest="run_dir", default="runs")
    parser.add_argument("--train-steps", dest="train_steps", type=int, default=1_000_000)
    parser.add_argument(
        "--reward-samples", dest="reward_samples", type=int,
        default=FBSF_DEFAULT_REWARD_SAMPLES,
    )
    parser.add_argument("--features", default=FBSF_DEFAULT_SF_FEATURES)
    parser.add_argument("--dataset-type", dest="dataset_type", default=EXORL_DATASET_TYPE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-episodes", dest="num_episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--execute", dest="dry_run", action="store_false")
    parser.add_argument("--manifest-only", dest="manifest_only", action="store_true")
    parser.add_argument("--eval-json", dest="eval_json", default=None)
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (compatible with ``fre.main`` baseline dispatch)."""
    parser = build_arg_parser()
    args, extra = parser.parse_known_args(argv)
    if extra:
        # Unknown flags are forwarded to the external trainer.
        pass

    config = FBSFConfig(
        method=args.method,
        domain=args.domain,
        task_set=args.task_set,
        repo_path=args.repo_path,
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        train_steps=args.train_steps,
        reward_samples=args.reward_samples,
        features=args.features,
        dataset_type=args.dataset_type,
        seeds=tuple(args.seeds),
        num_episodes=args.num_episodes,
        device=args.device,
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
        extra_train_args=tuple(extra),
    )
    runner = FBSFRunner(config)

    if args.manifest_only:
        manifest = runner.setup()
        print(json.dumps({k: v for k, v in manifest.items() if k != "commands"}, indent=2))
        return 0

    manifest = runner.run(progress=config.verbose)
    if args.eval_json:
        summary = runner.summary(args.eval_json)
        print(json.dumps(summary, indent=2))
    if manifest.get("status") == "external_repo_required":
        print(
            "[fb/sf] clone " + CONTROLLABLE_AGENT_URL + " and pass --repo-path to "
            "train/evaluate these baselines."
        )
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
