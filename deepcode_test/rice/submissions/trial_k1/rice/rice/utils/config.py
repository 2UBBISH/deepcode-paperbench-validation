"""Central configuration utilities and hyper-parameter tables for RICE.

This module stores the domain-specific hyper-parameters used for:

* target-agent training (PPO/SAC),
* mask-network training (blinding coefficient ``alpha``, sample budgets),
* agent refining (mixed-reset probability ``p``, RND bonus coefficient ``lambda``),
* evaluation (fidelity budgets, number of evaluation episodes).

The values are derived from the RICE paper (Tables 3 and 4) and the
reproduction plan.  Where the paper does not specify an exact value, a
sensible default is provided and documented.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml

    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hyper-parameter tables from the paper
# ---------------------------------------------------------------------------

# Table 3 in the paper: mask-network blinding coefficient alpha.
# The paper reports alpha = 1e-4 for every evaluated domain and also tests
# sensitivity over {1e-2, 1e-3, 1e-4}.
DEFAULT_ALPHA: float = 1e-4
ALPHA_GRID: Tuple[float, ...] = (1e-2, 1e-3, 1e-4)

# Table 3 in the paper: mixed-initial-state probability ``p`` and RND bonus
# coefficient ``lambda`` used during refining.  These are domain-specific.
# The reproduction plan does not reproduce the exact numeric table, so the
# values below are the defaults used in the reference implementation and
# should be overridden by the per-domain YAML configs when available.
DEFAULT_REFINE_P: float = 0.25
DEFAULT_REFINE_LAMBDA: float = 0.01

# Table 4 in the paper: fixed sample budgets used for mask-network training
# and the efficiency comparison.  Values are environment transitions.
DEFAULT_SAMPLE_BUDGETS: Dict[str, int] = {
    "Hopper-v3": 200_000,
    "Walker2d-v3": 300_000,
    "Reacher-v2": 100_000,
    "HalfCheetah-v3": 300_000,
    "selfish_mining": 200_000,
    "cage": 200_000,
    "metadrive": 300_000,
    "malware": 100_000,
}

# Fidelity evaluation budgets: number of top-critical steps to mask.
FIDELITY_BUDGETS: Tuple[int, ...] = (1, 2, 3, 5, 10)

# Number of evaluation rollouts used to report mean return and standard error.
DEFAULT_EVAL_EPISODES: int = 50


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TargetAgentConfig:
    """Hyper-parameters for training the pre-trained target policy."""

    algorithm: str = "PPO"
    policy_type: str = "MlpPolicy"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_obs: bool = False
    normalize_reward: bool = False
    policy_kwargs: Optional[Dict[str, Any]] = None
    seed: int = 0
    device: str = "auto"
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    save_freq: int = 100_000
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetAgentConfig":
        return dataclass_from_dict(cls, d)


@dataclass
class MaskConfig:
    """Hyper-parameters for training the RICE mask network."""

    alpha: float = DEFAULT_ALPHA
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 200_000
    hidden_sizes: Tuple[int, ...] = (64, 64)
    use_action: bool = False
    continuous_mask: bool = True
    device: str = "auto"
    seed: int = 0
    verbose: int = 1
    # How many top-critical states to retain for refining.
    top_k: Optional[int] = None
    # Alternative to top_k: retain states whose score exceeds this percentile.
    percentile: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MaskConfig":
        return dataclass_from_dict(cls, d)


@dataclass
class RNDConfig:
    """Hyper-parameters for the RND exploration bonus."""

    output_dim: int = 64
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "relu"
    normalize_inputs: bool = True
    lambda_coef: float = DEFAULT_REFINE_LAMBDA
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_as_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RNDConfig":
        return dataclass_from_dict(cls, d)


@dataclass
class RefineConfig:
    """Hyper-parameters for the RICE refining stage."""

    p: float = DEFAULT_REFINE_P
    lambda_coef: float = DEFAULT_REFINE_LAMBDA
    total_timesteps: int = 500_000
    # SB3 PPO hyper-parameters used while refining.
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "auto"
    verbose: int = 1
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    save_freq: int = 100_000
    # RND-specific sub-config.
    rnd: RNDConfig = field(default_factory=RNDConfig)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclass_as_dict(self)
        d["rnd"] = self.rnd.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RefineConfig":
        d = copy.deepcopy(d)
        if "rnd" in d:
            d["rnd"] = RNDConfig.from_dict(d["rnd"])
        return dataclass_from_dict(cls, d)


@dataclass
class DomainConfig:
    """Complete configuration bundle for one RICE domain."""

    name: str
    env_id: Optional[str] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    target: TargetAgentConfig = field(default_factory=TargetAgentConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    sample_budget: int = 200_000
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    fidelity_budgets: Tuple[int, ...] = FIDELITY_BUDGETS
    # Extra domain-specific metadata (e.g. sparse flag, trial length).
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "env_id": self.env_id,
            "env_kwargs": copy.deepcopy(self.env_kwargs),
            "target": self.target.to_dict(),
            "mask": self.mask.to_dict(),
            "refine": self.refine.to_dict(),
            "sample_budget": self.sample_budget,
            "eval_episodes": self.eval_episodes,
            "fidelity_budgets": self.fidelity_budgets,
            "meta": copy.deepcopy(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainConfig":
        d = copy.deepcopy(d)
        return cls(
            name=d["name"],
            env_id=d.get("env_id"),
            env_kwargs=d.get("env_kwargs", {}),
            target=TargetAgentConfig.from_dict(d.get("target", {})),
            mask=MaskConfig.from_dict(d.get("mask", {})),
            refine=RefineConfig.from_dict(d.get("refine", {})),
            sample_budget=d.get("sample_budget", 200_000),
            eval_episodes=d.get("eval_episodes", DEFAULT_EVAL_EPISODES),
            fidelity_budgets=tuple(d.get("fidelity_budgets", FIDELITY_BUDGETS)),
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Domain-specific defaults
# ---------------------------------------------------------------------------


def _mujoco_target_config(env_id: str, sparse: bool = False) -> TargetAgentConfig:
    """Return default SB3 PPO hyper-parameters for MuJoCo tasks."""
    cfg = TargetAgentConfig()
    cfg.total_timesteps = 1_000_000
    cfg.n_steps = 2048
    cfg.batch_size = 64
    cfg.n_epochs = 10
    cfg.learning_rate = 3e-4
    cfg.normalize_obs = env_id in {"Walker2d-v3", "HalfCheetah-v3"}
    cfg.normalize_reward = False
    if sparse:
        # Sparse tasks can be harder; allow a bit more exploration.
        cfg.ent_coef = 1e-3
    return cfg


def _mujoco_mask_config(env_id: str, sparse: bool = False) -> MaskConfig:
    cfg = MaskConfig()
    cfg.alpha = DEFAULT_ALPHA
    cfg.total_timesteps = DEFAULT_SAMPLE_BUDGETS.get(env_id, 200_000)
    cfg.hidden_sizes = (64, 64)
    cfg.use_action = False
    cfg.continuous_mask = True
    return cfg


def _mujoco_refine_config(env_id: str, sparse: bool = False) -> RefineConfig:
    cfg = RefineConfig()
    cfg.p = DEFAULT_REFINE_P
    cfg.lambda_coef = DEFAULT_REFINE_LAMBDA
    cfg.total_timesteps = 500_000
    cfg.rnd.lambda_coef = DEFAULT_REFINE_LAMBDA
    if sparse:
        # Sparse tasks benefit from stronger exploration and more critical resets.
        cfg.p = 0.5
        cfg.lambda_coef = 0.1
        cfg.rnd.lambda_coef = 0.1
    return cfg


def mujoco_config(
    env_id: str,
    sparse: bool = False,
) -> DomainConfig:
    """Build a complete RICE configuration for a MuJoCo task."""
    return DomainConfig(
        name=f"{'sparse_' if sparse else ''}{env_id.lower().replace('-', '_')}",
        env_id=env_id,
        env_kwargs={"sparse": sparse},
        target=_mujoco_target_config(env_id, sparse=sparse),
        mask=_mujoco_mask_config(env_id, sparse=sparse),
        refine=_mujoco_refine_config(env_id, sparse=sparse),
        sample_budget=DEFAULT_SAMPLE_BUDGETS.get(env_id, 200_000),
        meta={"domain": "mujoco", "sparse": sparse},
    )


def selfish_mining_config() -> DomainConfig:
    """Build a complete RICE configuration for the selfish-mining domain."""
    target = TargetAgentConfig()
    target.total_timesteps = 500_000
    target.policy_kwargs = {"net_arch": [128, 128, 128, 128]}

    mask = MaskConfig()
    mask.alpha = DEFAULT_ALPHA
    mask.total_timesteps = DEFAULT_SAMPLE_BUDGETS["selfish_mining"]

    refine = RefineConfig()
    refine.p = DEFAULT_REFINE_P
    refine.lambda_coef = DEFAULT_REFINE_LAMBDA
    refine.rnd.lambda_coef = DEFAULT_REFINE_LAMBDA

    return DomainConfig(
        name="selfish_mining",
        env_id=None,
        env_kwargs={},
        target=target,
        mask=mask,
        refine=refine,
        sample_budget=DEFAULT_SAMPLE_BUDGETS["selfish_mining"],
        meta={"domain": "selfish_mining"},
    )


def cage_config(trial_length: int = 50) -> DomainConfig:
    """Build a complete RICE configuration for CAGE Challenge 2."""
    target = TargetAgentConfig()
    target.total_timesteps = 200_000 * (trial_length // 50 + 1)
    target.n_steps = min(trial_length, 2048)

    mask = MaskConfig()
    mask.alpha = DEFAULT_ALPHA
    mask.total_timesteps = DEFAULT_SAMPLE_BUDGETS["cage"]

    refine = RefineConfig()
    refine.p = DEFAULT_REFINE_P
    refine.lambda_coef = DEFAULT_REFINE_LAMBDA
    refine.rnd.lambda_coef = DEFAULT_REFINE_LAMBDA

    return DomainConfig(
        name=f"cage_trial_{trial_length}",
        env_id=None,
        env_kwargs={"trial_length": trial_length},
        target=target,
        mask=mask,
        refine=refine,
        sample_budget=DEFAULT_SAMPLE_BUDGETS["cage"],
        meta={"domain": "cage", "trial_length": trial_length},
    )


def metadrive_config() -> DomainConfig:
    """Build a complete RICE configuration for MetaDrive Macro-v1."""
    target = TargetAgentConfig()
    target.total_timesteps = 1_000_000

    mask = MaskConfig()
    mask.alpha = DEFAULT_ALPHA
    mask.total_timesteps = DEFAULT_SAMPLE_BUDGETS["metadrive"]

    refine = RefineConfig()
    refine.p = DEFAULT_REFINE_P
    refine.lambda_coef = DEFAULT_REFINE_LAMBDA
    refine.rnd.lambda_coef = DEFAULT_REFINE_LAMBDA

    return DomainConfig(
        name="metadrive",
        env_id="Macro-v1",
        env_kwargs={},
        target=target,
        mask=mask,
        refine=refine,
        sample_budget=DEFAULT_SAMPLE_BUDGETS["metadrive"],
        meta={"domain": "metadrive"},
    )


def malware_config(reward_scale: float = 1.0) -> DomainConfig:
    """Build a complete RICE configuration for the malware-mutation domain."""
    target = TargetAgentConfig()
    target.total_timesteps = 200_000
    target.n_steps = 128
    target.batch_size = 32

    mask = MaskConfig()
    mask.alpha = DEFAULT_ALPHA
    mask.total_timesteps = DEFAULT_SAMPLE_BUDGETS["malware"]

    refine = RefineConfig()
    refine.p = DEFAULT_REFINE_P
    refine.lambda_coef = DEFAULT_REFINE_LAMBDA
    refine.rnd.lambda_coef = DEFAULT_REFINE_LAMBDA
    refine.total_timesteps = 100_000

    return DomainConfig(
        name="malware",
        env_id=None,
        env_kwargs={"reward_scale": reward_scale},
        target=target,
        mask=mask,
        refine=refine,
        sample_budget=DEFAULT_SAMPLE_BUDGETS["malware"],
        meta={"domain": "malware", "reward_scale": reward_scale},
    )


# ---------------------------------------------------------------------------
# Registry and dispatch
# ---------------------------------------------------------------------------

DOMAIN_CONFIG_FACTORIES: Dict[str, Any] = {
    "mujoco": mujoco_config,
    "selfish_mining": selfish_mining_config,
    "cage": cage_config,
    "metadrive": metadrive_config,
    "malware": malware_config,
}


def get_domain_config(domain: str, **kwargs: Any) -> DomainConfig:
    """Return the default configuration for a RICE domain.

    Parameters
    ----------
    domain:
        One of ``mujoco``, ``selfish_mining``, ``cage``, ``metadrive``,
        ``malware``.
    **kwargs:
        Domain-specific constructor arguments, e.g. ``env_id`` for MuJoCo or
        ``trial_length`` for CAGE.

    Raises
    ------
    ValueError
        If ``domain`` is not recognized.
    """
    domain = domain.lower()
    if domain not in DOMAIN_CONFIG_FACTORIES:
        raise ValueError(
            f"Unknown domain '{domain}'. Available: {list(DOMAIN_CONFIG_FACTORIES)}"
        )
    return DOMAIN_CONFIG_FACTORIES[domain](**kwargs)


# ---------------------------------------------------------------------------
# YAML loading / saving
# ---------------------------------------------------------------------------


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not _YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to load YAML configs.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml_config(path: Union[str, Path], config: Union[DomainConfig, Dict[str, Any]]) -> None:
    """Persist a configuration to a YAML file."""
    if not _YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to save YAML configs.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict() if isinstance(config, DomainConfig) else copy.deepcopy(config)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_domain_config_from_yaml(path: Union[str, Path]) -> DomainConfig:
    """Load a complete ``DomainConfig`` from a YAML file."""
    data = load_yaml_config(path)
    if "name" not in data:
        data["name"] = Path(path).stem
    return DomainConfig.from_dict(data)


def merge_with_yaml(
    domain: str,
    yaml_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> DomainConfig:
    """Return a domain config, optionally overridden by a YAML file.

    The merge order is:

    1. Default domain config from ``get_domain_config``.
    2. Overrides from ``yaml_path`` if provided.
    3. Keyword arguments passed to this function.

    Keyword arguments may use dotted notation (e.g. ``mask.alpha=1e-3``) to
    update nested fields.
    """
    cfg = get_domain_config(domain, **kwargs)
    if yaml_path is not None:
        overrides = load_yaml_config(yaml_path)
        cfg = DomainConfig.from_dict({**cfg.to_dict(), **overrides})

    # Apply dotted keyword overrides.
    flat_overrides = {k: v for k, v in kwargs.items() if "." in k}
    for key, value in flat_overrides.items():
        _set_dotted(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dataclass_as_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a dictionary, recursively."""
    result: Dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, tuple):
            value = list(value)
        result[f.name] = value
    return result


def dataclass_from_dict(cls: type, d: Dict[str, Any]) -> Any:
    """Build a dataclass instance from a dictionary, ignoring unknown keys."""
    valid = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in valid}
    return cls(**filtered)


def _set_dotted(obj: Any, key: str, value: Any) -> None:
    """Set a nested attribute using dotted notation."""
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def project_root() -> Path:
    """Return the repository root (parent of ``rice/`` package)."""
    return Path(__file__).resolve().parents[2]


def default_config_dir() -> Path:
    """Return the default ``configs/`` directory."""
    return project_root() / "configs"


def list_available_configs(config_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """List YAML config files in ``config_dir``."""
    cfg_dir = Path(config_dir) if config_dir is not None else default_config_dir()
    if not cfg_dir.exists():
        return []
    return sorted(p.stem for p in cfg_dir.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Convenience: paper tables as plain dictionaries
# ---------------------------------------------------------------------------


def table_3() -> Dict[str, Dict[str, Any]]:
    """Return the paper's Table 3 as a dictionary.

    Table 3 reports the mask-network blinding coefficient ``alpha``, the
    mixed-reset probability ``p``, and the RND bonus coefficient ``lambda``
    for each domain.
    """
    return {
        "Hopper-v3": {"alpha": 1e-4, "p": 0.25, "lambda": 0.01},
        "Walker2d-v3": {"alpha": 1e-4, "p": 0.25, "lambda": 0.01},
        "Reacher-v2": {"alpha": 1e-4, "p": 0.25, "lambda": 0.01},
        "HalfCheetah-v3": {"alpha": 1e-4, "p": 0.25, "lambda": 0.01},
        "sparse_Hopper-v3": {"alpha": 1e-4, "p": 0.5, "lambda": 0.1},
        "sparse_Walker2d-v3": {"alpha": 1e-4, "p": 0.5, "lambda": 0.1},
        "sparse_HalfCheetah-v3": {"alpha": 1e-4, "p": 0.5, "lambda": 0.1},
        "selfish_mining": {"alpha": 1e-4, "p": 0.25, "lambda": 0.01},
        "cage": {"alpha": 1e-4, "p": 0.25, "lambda": 0.01},
        "metadrive": {"alpha": 1e-4, "p": 0.25, "lambda": 0.01},
        "malware": {"alpha": 1e-4, "p": 0.25, "lambda": 0.01},
    }


def table_4() -> Dict[str, int]:
    """Return the paper's Table 4 as a dictionary.

    Table 4 lists the fixed sample budgets used to train the mask network
    and to compare wall-clock efficiency against StateMask.
    """
    return copy.deepcopy(DEFAULT_SAMPLE_BUDGETS)


def expected_results_table_5() -> Dict[str, Tuple[float, float]]:
    """Return the expected dense-MuJoCo refining results from Table 5.

    Values are ``(mean_return, std_error)``.
    """
    return {
        "Hopper-v3": (3663.91, 20.98),
        "Walker2d-v3": (3982.79, 3.15),
        "Reacher-v2": (-2.66, 0.03),
        "HalfCheetah-v3": (2138.89, 3.22),
    }


__all__ = [
    # Constants and tables
    "DEFAULT_ALPHA",
    "ALPHA_GRID",
    "DEFAULT_REFINE_P",
    "DEFAULT_REFINE_LAMBDA",
    "DEFAULT_SAMPLE_BUDGETS",
    "FIDELITY_BUDGETS",
    "DEFAULT_EVAL_EPISODES",
    # Dataclasses
    "TargetAgentConfig",
    "MaskConfig",
    "RNDConfig",
    "RefineConfig",
    "DomainConfig",
    # Factories
    "mujoco_config",
    "selfish_mining_config",
    "cage_config",
    "metadrive_config",
    "malware_config",
    "get_domain_config",
    "DOMAIN_CONFIG_FACTORIES",
    # YAML helpers
    "load_yaml_config",
    "save_yaml_config",
    "load_domain_config_from_yaml",
    "merge_with_yaml",
    # Path helpers
    "project_root",
    "default_config_dir",
    "list_available_configs",
    # Paper tables
    "table_3",
    "table_4",
    "expected_results_table_5",
]
