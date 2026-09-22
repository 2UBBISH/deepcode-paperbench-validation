"""PPO fine-tuning baseline for RICE (Sec. 4.1 "Baseline Refining Methods").

The paper's first refining baseline is:

    "PPO fine-tuning" (Schulman et al., 2017), i.e., lowering the learning rate
    and continuing training with the PPO algorithm.

Appendix C.1 only notes that baselines use the released code of the authors (or
our own implementation when no code is released); everything else about this
baseline follows the Sec. 4.1 description above.  Concretely this baseline
continues *plain* PPO training from the frozen pre-trained policy :math:`\\pi`
with **no** explanation-driven machinery:

* the initial state distribution is the environment's default :math:`\\rho(s)`
  (no mixed initial state distribution, i.e. ``p = 0`` / ``use_mixed_init=False``);
* there is **no** resetting to mask-identified critical states;
* there is no RND intrinsic reward (:math:`\\lambda = 0`);
* the learning rate is *lowered* relative to the pre-training learning rate.

The implementation deliberately reuses :class:`rice.refining.ppo_refine.PPORefiner`
so that the only differences between RICE and this baseline are exactly the
paper's claimed contributions (mixed initial state distribution + exploration
bonus).  That keeps the comparison fair and isolates the effect of the
refinement mechanism.

The paper reports (Sec. 4.3, Table 1) that PPO fine-tuning yields only
marginal improvements over the "No Refine" pre-trained policy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np  # noqa: F401  (kept for API parity / downstream use)

# ---------------------------------------------------------------------------
# Defensive imports: the module must stay importable even when the (torch-heavy)
# refining engine is unavailable, so every project import degrades gracefully.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from ..refining.ppo_refine import (  # type: ignore
        DEFAULT_BATCH_SIZE,
        DEFAULT_CLIP_RANGE,
        DEFAULT_ENT_COEF,
        DEFAULT_GAE_LAMBDA,
        DEFAULT_GAMMA,
        DEFAULT_HORIZON,
        DEFAULT_MAX_GRAD_NORM,
        DEFAULT_N_EPOCHS,
        DEFAULT_PPO_LR,
        DEFAULT_VF_COEF,
        PPORefiner,
        RefinePPOConfig,
        evaluate_refined_policy,
        make_refiner,
        unpack_reset,
    )

    _HAS_REFINER = True
except Exception:  # pragma: no cover - fallback definitions
    _HAS_REFINER = False

    DEFAULT_HORIZON = 1000
    DEFAULT_PPO_LR = 3e-4
    DEFAULT_GAMMA = 0.99
    DEFAULT_GAE_LAMBDA = 0.95
    DEFAULT_CLIP_RANGE = 0.2
    DEFAULT_N_EPOCHS = 10
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_VF_COEF = 0.5
    DEFAULT_ENT_COEF = 0.0
    DEFAULT_MAX_GRAD_NORM = 0.5

    PPORefiner = None  # type: ignore
    RefinePPOConfig = None  # type: ignore
    make_refiner = None  # type: ignore
    evaluate_refined_policy = None  # type: ignore
    unpack_reset = None  # type: ignore

try:  # pragma: no cover - import guard
    from ..models.policies import save_policy  # type: ignore

    _HAS_POLICIES = True
except Exception:  # pragma: no cover
    _HAS_POLICIES = False
    save_policy = None  # type: ignore


try:  # pragma: no cover - import guard
    from ..utils.io import ensure_dir  # type: ignore
except Exception:  # pragma: no cover

    def ensure_dir(path: str) -> str:  # type: ignore
        os.makedirs(path, exist_ok=True)
        return path


try:  # pragma: no cover - import guard
    from ..utils.logging import get_logger  # type: ignore
except Exception:  # pragma: no cover

    def get_logger(name: str = "rice", out_dir: Optional[str] = None, **kwargs):  # type: ignore
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        return logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The paper says the PPO fine-tuning baseline *lowers* the learning rate.
#: With SB3's default PPO lr = 3e-4 for pre-training we use 1e-4 for fine-tuning.
DEFAULT_FINETUNE_LR = 1e-4
#: Alternative, even gentler fine-tuning rate (kept for sensitivity runs).
DEFAULT_FINETUNE_LR_LOW = 3e-5
#: Default number of refinement environment steps (only trends are required).
DEFAULT_FINETUNE_TIMESTEPS = 200_000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOFinetuneConfig:
    """Hyper-parameters for the PPO fine-tuning baseline.

    Only trends are reproduced (per the reproduction plan), so unspecified
    numbers fall back to the Stable-Baselines3 PPO defaults also used by RICE.
    """

    env_id: str = "default"
    # -- fine-tuning specifics -------------------------------------------------
    lr: float = DEFAULT_FINETUNE_LR
    finetune_lr: Optional[float] = None          # alias, wins over ``lr``
    total_timesteps: int = DEFAULT_FINETUNE_TIMESTEPS
    total_iterations: Optional[int] = None
    # -- standard PPO (SB3 defaults, identical to RICE) ------------------------
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_range: float = DEFAULT_CLIP_RANGE
    n_epochs: int = DEFAULT_N_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    vf_coef: float = DEFAULT_VF_COEF
    ent_coef: float = DEFAULT_ENT_COEF
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    n_steps: Optional[int] = None
    target_kl: Optional[float] = None
    normalize_advantage: bool = True
    # -- deliberately DISABLED RICE components --------------------------------
    use_mixed_init: bool = False
    use_rnd: bool = False
    p: float = 0.0
    lam: float = 0.0
    # -- bookkeeping -----------------------------------------------------------
    device: str = "cpu"
    seed: Optional[int] = None
    copy_policy: bool = True
    deterministic_eval: bool = True
    eval_episodes: int = 10

    def __post_init__(self) -> None:
        if self.finetune_lr is not None:
            self.lr = float(self.finetune_lr)
        # The baseline never uses the RICE mechanisms; force them off so a
        # shared YAML config cannot silently turn this baseline into RICE.
        self.use_mixed_init = False
        self.use_rnd = False
        self.p = 0.0
        self.lam = 0.0

    @property
    def learning_rate(self) -> float:
        return float(self.lr)

    @property
    def timesteps(self) -> int:
        return int(self.total_timesteps)

    # -- (de)serialisation ----------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "PPOFinetuneConfig":
        cfg = dict(cfg or {})
        # Accept nested sections from the shared YAML files.
        for section in ("ppo_finetune", "finetune", "ppo", "refine", "baseline"):
            sub = cfg.get(section)
            if isinstance(sub, dict):
                merged = dict(sub)
                merged.update({k: v for k, v in cfg.items() if k != section})
                cfg = merged
                break
        alias = {
            "lr": ("learning_rate",),
            "finetune_lr": ("finetune_lr", "fine_tune_lr", "finetuning_lr"),
            "total_timesteps": ("timesteps", "n_timesteps", "samples"),
            "total_iterations": ("iterations", "n_iterations"),
            "lam": ("lambda", "lambda_", "rnd_lambda"),
            "p": ("beta", "prob", "probability"),
            "clip_range": ("clip",),
            "gamma": ("discount", "discount_factor"),
            "n_steps": ("rollout_length", "T"),
            "n_epochs": ("n_epoch", "epochs", "update_epochs"),
            "copy_policy": ("copy", "clone_policy"),
        }
        for target, names in alias.items():
            for name in names:
                if cfg.get(name) is not None:
                    cfg.setdefault(target, cfg[name])
                    break
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in cfg.items() if k in known and v is not None}
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["learning_rate"] = self.learning_rate
        return out

    # -- conversion into the shared refinement config -------------------------
    def to_refine_config(self) -> Any:
        """Build a :class:`RefinePPOConfig` with the RICE mechanisms disabled."""
        payload = {
            "lr": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "n_steps": self.n_steps,
            "target_kl": self.target_kl,
            "normalize_advantage": self.normalize_advantage,
            "use_mixed_init": False,
            "use_rnd": False,
            "p": 0.0,
            "lam": 0.0,
            "device": self.device,
            "seed": self.seed,
            "copy_policy": self.copy_policy,
            "env_id": self.env_id,
        }
        if RefinePPOConfig is not None:
            try:
                return RefinePPOConfig.from_dict(payload)
            except Exception:  # pragma: no cover - defensive
                pass
        return payload


# ---------------------------------------------------------------------------
# The finetuner
# ---------------------------------------------------------------------------


class PPOFinetuner(PPORefiner if PPORefiner is not None else object):  # type: ignore[misc]
    """Continue plain PPO training from the frozen pre-trained policy.

    Functionally this is :class:`PPORefiner` with (i) the mixed initial state
    distribution off (``p = 0``, always reset from :math:`\\rho`), (ii) the RND
    exploration bonus off (``\\lambda = 0``) and (iii) a *lowered* learning rate.
    """

    name = "PPO-Finetune"

    def __init__(
        self,
        env: Any,
        policy: Any = None,
        config: Optional[Any] = None,
        env_id: str = "default",
        device: str = "cpu",
        logger: Any = None,
        lr: Optional[float] = None,
        total_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if PPORefiner is None:  # pragma: no cover - hard failure path
            raise ImportError(
                "rice.refining.ppo_refine is required for the PPO fine-tuning baseline"
            )

        cfg = self._coerce_config(config)
        cfg.env_id = env_id or cfg.env_id
        cfg.device = device or cfg.device
        if seed is not None:
            cfg.seed = seed
        if lr is not None:
            cfg.lr = float(lr)
        if total_timesteps is not None:
            cfg.total_timesteps = int(total_timesteps)

        self.finetune_config = cfg

        super().__init__(
            env,
            policy=policy,
            config=cfg.to_refine_config(),
            env_id=cfg.env_id,
            device=cfg.device,
            logger=logger,
            **kwargs,
        )
        # Guarantee the disabled components even if the parent re-enabled them.
        self.sampler = None
        self.rnd = None
        self.mask_net = None
        self.config.use_mixed_init = False
        self.config.use_rnd = False
        try:
            self.config.p = 0.0
            self.config.lam = 0.0
        except Exception:  # pragma: no cover - defensive
            pass
        self.logger = logger or get_logger("rice.baselines.ppo_finetune")

    # -- helpers --------------------------------------------------------------
    @classmethod
    def _coerce_config(cls, config: Optional[Any]) -> PPOFinetuneConfig:
        if config is None:
            return PPOFinetuneConfig()
        if isinstance(config, PPOFinetuneConfig):
            return config
        if isinstance(config, dict):
            return PPOFinetuneConfig.from_dict(config)
        # RefinePPOConfig or any other object: copy the relevant fields over.
        data: Dict[str, Any] = {}
        for fname in PPOFinetuneConfig.__dataclass_fields__:  # type: ignore[attr-defined]
            if hasattr(config, fname):
                data[fname] = getattr(config, fname)
        return PPOFinetuneConfig(**data)

    def sample_initial_state(self, policy: Any = None, reset_kwargs: Optional[Dict] = None):
        """Always sample :math:`s_0 \\sim \\rho(s)` (no critical-state reset)."""
        kwargs = dict(reset_kwargs or {})
        if unpack_reset is not None:
            obs, info = unpack_reset(self.env.reset(**kwargs))
        else:  # pragma: no cover - defensive
            result = self.env.reset(**kwargs)
            obs, info = result if isinstance(result, tuple) and len(result) == 2 else (result, {})
        return obs, info

    def will_reset_to_critical(self) -> bool:
        """The baseline never resets to the mask-identified critical state."""
        return False

    def set_p(self, p: float) -> float:  # pragma: no cover - API parity
        """Ignored: the fine-tuning baseline has no mixed initial distribution."""
        self.config.p = 0.0
        return 0.0

    def set_lambda(self, lam: float) -> float:  # pragma: no cover - API parity
        """Ignored: the fine-tuning baseline has no exploration bonus."""
        self.config.lam = 0.0
        return 0.0

    def summary(self) -> Dict[str, Any]:
        try:
            base: Dict[str, Any] = super().summary()  # type: ignore[misc]
        except Exception:  # pragma: no cover - defensive
            base = {}
        base.update(
            {
                "method": self.name,
                "env_id": self.finetune_config.env_id,
                "lr": float(self.finetune_config.lr),
                "use_mixed_init": False,
                "use_rnd": False,
                "p": 0.0,
                "lam": 0.0,
                "total_timesteps": int(self.finetune_config.total_timesteps),
            }
        )
        return base

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        if save_policy is None or self.policy is None:  # pragma: no cover
            raise RuntimeError("save_policy unavailable")
        meta = {"method": self.name, "baseline": "ppo_finetune"}
        if extra:
            meta.update(extra)
        return save_policy(self.policy, path, env_id=self.finetune_config.env_id, extra=meta)


# ---------------------------------------------------------------------------
# Functional entry points
# ---------------------------------------------------------------------------


def make_finetuner(
    env: Any,
    policy: Any = None,
    config: Optional[Any] = None,
    env_id: str = "default",
    **kwargs: Any,
) -> PPOFinetuner:
    """Factory mirroring :func:`rice.refining.ppo_refine.make_refiner`."""
    if isinstance(config, str):  # allow ``make_finetuner(env, policy, "hopper")``
        env_id, config = config, None
    return PPOFinetuner(env, policy=policy, config=config, env_id=env_id, **kwargs)


build_finetuner = make_finetuner


def ppo_finetune_policy(
    env: Any,
    policy: Any = None,
    total_timesteps: Optional[int] = None,
    total_iterations: Optional[int] = None,
    env_id: str = "default",
    config: Optional[Any] = None,
    logger: Any = None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress: bool = False,
    evaluate: bool = False,
    eval_episodes: int = 10,
    lr: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[Any, PPOFinetuner]:
    """Run the PPO fine-tuning baseline end-to-end.

    Returns ``(refined_policy, finetuner)`` so experiment drivers can read back
    both the refined policy and the logged history (the Table-1
    "PPO fine-tuning" rows).
    """
    finetuner = make_finetuner(
        env,
        policy=policy,
        config=config,
        env_id=env_id,
        device=device,
        logger=logger,
        lr=lr,
        total_timesteps=total_timesteps,
        seed=seed,
        **kwargs,
    )
    finetuner.train(
        total_timesteps=finetuner.finetune_config.total_timesteps,
        total_iterations=total_iterations or finetuner.finetune_config.total_iterations,
        logger=logger,
        progress=progress,
    )
    if evaluate and evaluate_refined_policy is not None:
        try:
            finetuner.eval_history.append(  # type: ignore[attr-defined]
                evaluate_refined_policy(
                    env,
                    finetuner.policy,
                    env_id=env_id,
                    n_episodes=eval_episodes,
                    deterministic=True,
                    device=device,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            if logger is not None:
                logger.warning("PPO fine-tuning evaluation failed: %s", exc)
    if save_path and save_policy is not None and finetuner.policy is not None:
        ensure_dir(os.path.dirname(os.path.abspath(save_path)) or ".")
        finetuner.save(save_path)
    return finetuner.policy, finetuner


# Common aliases used by experiment drivers / registries.
ppofinetune = ppo_finetune_policy
train_ppo_finetune = ppo_finetune_policy
finetune = ppo_finetune_policy
PPOFinetuneBaseline = PPOFinetuner


def describe_ppo_finetune(finetuner: Optional[PPOFinetuner] = None) -> str:
    """One-line human-readable description used in logs / tables."""
    if finetuner is None:
        return "PPO fine-tuning baseline (lowered lr, continue PPO from rho, no RND)"
    cfg = finetuner.finetune_config
    return (
        f"PPO fine-tuning baseline [{cfg.env_id}] lr={cfg.lr:g}, "
        f"timesteps={cfg.total_timesteps}, mixed_init=off, rnd=off"
    )


__all__ = [
    "PPOFinetuneConfig",
    "PPOFinetuner",
    "PPOFinetuneBaseline",
    "DEFAULT_FINETUNE_LR",
    "DEFAULT_FINETUNE_LR_LOW",
    "DEFAULT_FINETUNE_TIMESTEPS",
    "make_finetuner",
    "build_finetuner",
    "ppo_finetune_policy",
    "ppofinetune",
    "train_ppo_finetune",
    "finetune",
    "describe_ppo_finetune",
]
