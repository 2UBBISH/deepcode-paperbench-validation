"""RoboticSequence (Meta-World) pre-training + fine-tuning driver.

This module glues together

* :mod:`src.robotic_sequence.env`      -- Algorithm 1 sequential stage wrapper
* :mod:`src.robotic_sequence.sac`      -- SAC learner with per-stage heads
* :mod:`src.retention.*`               -- actor-only knowledge retention losses

and reproduces the pipeline described in Section 3 / Appendix B.3 of
Wołczyk et al. (2024), "Fine-tuning Reinforcement Learning Models is Secretly a
Forgetting Mitigation Problem":

1. **Pre-training** of ``pi_*`` on a subset of the stages.  The paper's main
   results use a ``pi_*`` trained on the *last two* stages (``peg-unplug-side``,
   ``push-wall``), which are the FAR stages of the main sequence.  The addendum
   additionally mentions an all-stages pre-training variant, exposed here as
   ``pretrain.scope = all_stages``.
2. **Fine-tuning** on the full RoboticSequence with one of four settings:
   ``none`` (vanilla), ``ewc``, ``bc``, ``em`` (episodic memory).  Retention is
   applied to the **actor only** -- the critic coefficient is always ``0``.
3. **Evaluation**: per-stage success rates (Figure 7), forward transfer over
   prefix-task lengths (Table 6), and expert-action log-likelihood of the
   ``push-wall`` task every ``finetune.loglikelihood_every`` steps (Figure 8).

The default configuration is :file:`configs/robotic_sequence.yaml`.  Running
without Meta-World installed is supported through the stub environment
(``env.stub: true``), which allows cheap CPU smoke tests of the whole pipeline.

CLI::

    python -m src.robotic_sequence.train_robotic --config configs/robotic_sequence.yaml
    python -m src.robotic_sequence.train_robotic --config configs/robotic_sequence.yaml \\
        --set retention.method=bc --set seed=1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Soft imports of the project's own modules (kept import-safe for docs/tests).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import shim
    from src.common.config import Config, apply_overrides, dump_config, load_config
except Exception:  # pragma: no cover
    from ..common.config import Config, apply_overrides, dump_config, load_config  # type: ignore

try:  # pragma: no cover
    from src.common.logging_utils import MetricLogger, get_logger, human_format, log_config
except Exception:  # pragma: no cover
    from ..common.logging_utils import MetricLogger, get_logger, human_format, log_config  # type: ignore

try:  # pragma: no cover
    from src.common.seeding import make_generator, seed_env, set_seed
except Exception:  # pragma: no cover
    from ..common.seeding import make_generator, seed_env, set_seed  # type: ignore

try:  # pragma: no cover
    from src.common.checkpointing import ensure_dir, latest_checkpoint, save_checkpoint
except Exception:  # pragma: no cover
    from ..common.checkpointing import ensure_dir, latest_checkpoint, save_checkpoint  # type: ignore

try:  # pragma: no cover
    from src.robotic_sequence.env import (
        ALTERNATIVE_ORDERINGS,
        BETA,
        CLOSE_TASKS,
        FAR_TASKS,
        ROBOTIC_SEQUENCE_TASKS,
        TIME_LIMIT,
        RoboticSequenceEnv,
        forward_transfer_metric,
        make_stage_env,
        per_stage_success_rate,
        tasks_for,
    )
except Exception:  # pragma: no cover
    from .env import (  # type: ignore
        ALTERNATIVE_ORDERINGS,
        BETA,
        CLOSE_TASKS,
        FAR_TASKS,
        ROBOTIC_SEQUENCE_TASKS,
        TIME_LIMIT,
        RoboticSequenceEnv,
        forward_transfer_metric,
        make_stage_env,
        per_stage_success_rate,
        tasks_for,
    )

try:  # pragma: no cover
    from src.robotic_sequence.sac import (
        ReplayBuffer,
        SACAgent,
        SACConfig,
        build_sac_agent,
    )
except Exception:  # pragma: no cover
    from .sac import (  # type: ignore
        ReplayBuffer,
        SACAgent,
        SACConfig,
        build_sac_agent,
    )

try:  # pragma: no cover
    from src.retention.ewc import EWC
except Exception:  # pragma: no cover
    EWC = None  # type: ignore

try:  # pragma: no cover
    from src.retention.behavioral_cloning import BehavioralCloning, build_bc_buffer
except Exception:  # pragma: no cover
    BehavioralCloning = None  # type: ignore
    build_bc_buffer = None  # type: ignore

try:  # pragma: no cover
    from src.retention.episodic_memory import EpisodicMemory
except Exception:  # pragma: no cover
    EpisodicMemory = None  # type: ignore

try:  # pragma: no cover
    from src.retention.fisher import FisherEstimator
except Exception:  # pragma: no cover
    FisherEstimator = None  # type: ignore


LOGGER = get_logger("robotic_sequence.train")


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Fetch a dotted ``a.b.c`` value from a ``Config`` / mapping / object."""
    obj = cfg
    for key in path.split("."):
        if obj is None:
            return default
        if isinstance(obj, dict) or hasattr(obj, "get"):
            try:
                if key in obj:  # type: ignore[operator]
                    obj = obj[key]  # type: ignore[index]
                    continue
            except TypeError:
                pass
            obj = getattr(obj, key, None)
        else:
            obj = getattr(obj, key, default)
    return default if obj is None else obj


def resolve_task_order(cfg: Any) -> Tuple[str, ...]:
    """Resolve the stage sequence from the config (``env.tasks`` / ``env.task_order``)."""
    tasks = _cfg_get(cfg, "env.tasks", None)
    if tasks:
        if isinstance(tasks, str):
            return tuple(t.strip() for t in tasks.split(",") if t.strip())
        return tuple(tasks)
    order = _cfg_get(cfg, "env.task_order", "main")
    return tuple(tasks_for(order))


def close_far_split(tasks: Iterable[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split a stage list into (CLOSE, FAR) tasks following §B.3."""
    close, far = [], []
    for task in tasks:
        base = task.split("-v")[0]
        (far if base in FAR_TASKS else close).append(task)
    return tuple(close), tuple(far)


def _retention_method(cfg: Any) -> str:
    method = _cfg_get(cfg, "retention.method", "none")
    return (method or "none").lower()


# --------------------------------------------------------------------------- #
# Pre-training of pi_*
# --------------------------------------------------------------------------- #
@dataclass
class PretrainResult:
    """Container for the outcome of ``pi_*`` pre-training."""

    tasks: Tuple[str, ...]
    steps: int
    success_rates: Dict[str, float]
    checkpoint: Optional[str] = None
    history: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tasks": list(self.tasks),
            "steps": self.steps,
            "success_rates": dict(self.success_rates),
            "checkpoint": self.checkpoint,
        }


def pretrain_tasks_from_scope(tasks: Sequence[str], scope: str = "last_two") -> Tuple[str, ...]:
    """Return the stage subset used to pre-train ``pi_*``.

    ``last_two``  -> the last two stages of the sequence (the FAR stages of the
    main ordering, i.e. ``peg-unplug-side`` and ``push-wall``), which is the
    protocol used for the paper's main RoboticSequence results.

    ``all_stages`` -> the whole sequence (addendum / Continual World variant),
    kept as a baseline.
    """
    scope = (scope or "last_two").lower()
    if scope in ("all", "all_stages", "full"):
        return tuple(tasks)
    if scope in ("first_two", "close_two"):
        return tuple(tasks[:2])
    return tuple(tasks[-2:])


def _build_env(cfg: Any, task_order: Sequence[str], stub: bool, seed: int) -> RoboticSequenceEnv:
    env = RoboticSequenceEnv(
        task_order=tuple(task_order),
        time_limit=int(_cfg_get(cfg, "env.time_limit", TIME_LIMIT)),
        beta=float(_cfg_get(cfg, "env.beta", BETA)),
        seed=seed,
        stub=bool(stub),
        append_timestep=bool(_cfg_get(cfg, "env.append_timestep", True)),
        append_stage_onehot=bool(_cfg_get(cfg, "env.append_stage_onehot", False)),
        terminal_on_time_limit=bool(_cfg_get(cfg, "env.terminal_on_time_limit", True)),
        solve_probability=float(_cfg_get(cfg, "env.solve_probability", 1.0)),
    )
    return env


def pretrain_pistar(
    cfg: Any,
    *,
    stub: bool = False,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    num_steps: Optional[int] = None,
    tasks: Optional[Sequence[str]] = None,
    checkpoint: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[SACAgent, PretrainResult]:
    """Pre-train ``pi_*`` on ``tasks`` with SAC (Algorithm 1, per-stage heads).

    Returns the trained agent together with a :class:`PretrainResult`.  The
    agent is left in ``eval`` mode and its actor snapshot represents ``pi_*``.
    """
    seed = int(_cfg_get(cfg, "seed", 0) if seed is None else seed)
    set_seed(seed)
    device = device or str(_cfg_get(cfg, "compute.device", "cpu"))
    scope = str(_cfg_get(cfg, "pretrain.scope", "last_two"))
    seq = tuple(tasks) if tasks else resolve_task_order(cfg)
    pretrain_tasks = tuple(tasks) if tasks else pretrain_tasks_from_scope(seq, scope)
    num_steps = int(num_steps or _cfg_get(cfg, "pretrain.num_steps", 500_000))

    env = _build_env(cfg, pretrain_tasks, stub=stub, seed=seed)
    obs_dim = int(env.observation_dim)
    action_dim = int(env.action_dim)
    n_stages = int(env.n_stages)

    agent = build_sac_agent(
        cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_stages=n_stages,
        device=device,
        retention=None,
        seed=seed,
    )

    batch_size = int(_cfg_get(cfg, "sac.batch_size", 128))
    update_every = int(_cfg_get(cfg, "sac.update_every", 1))
    updates_per_step = int(_cfg_get(cfg, "sac.updates_per_step", 1))
    warmup = int(_cfg_get(cfg, "sac.warmup_steps", 1000))
    eval_every = int(_cfg_get(cfg, "pretrain.eval_every", max(num_steps // 10, 1)))
    eval_episodes = int(_cfg_get(cfg, "pretrain.eval_episodes", 20))
    log_every = int(_cfg_get(cfg, "finetune.log_every", 10_000))

    rng = make_generator(seed)
    obs = env.reset(seed=seed)
    ep_return, ep_steps = 0.0, 0
    history: List[Dict[str, float]] = []

    for step in range(1, num_steps + 1):
        if step <= warmup:
            action = np.asarray(
                [rng.uniform(-1.0, 1.0) for _ in range(action_dim)], dtype=np.float32
            )
        else:
            action = agent.select_action(
                obs, stage_id=env.current_task.index, deterministic=False
            )
        action = np.asarray(action, dtype=np.float32).reshape(action_dim)

        next_obs, reward, done, info = env.step(action)
        agent.replay_buffer.add(
            {
                "obs": obs,
                "action": action,
                "reward": float(reward),
                "next_obs": next_obs,
                "done": bool(done),
                "stage_id": int(info.get("stage_id", env.current_task.index)),
            }
        ) if getattr(agent, "replay_buffer", None) is not None else None

        ep_return += float(reward)
        ep_steps += 1

        if step > warmup and step % max(update_every, 1) == 0:
            for _ in range(max(updates_per_step, 1)):
                if len(agent.replay_buffer) >= batch_size:
                    agent.update(batch_size=batch_size) if "batch_size" in (
                        agent.update.__code__.co_varnames  # type: ignore[attr-defined]
                    ) else agent.update(agent.replay_buffer.sample(batch_size, rng, device))

        if done:
            obs = env.reset()
            ep_return, ep_steps = 0.0, 0
        else:
            obs = next_obs

        if step % log_every == 0:
            LOGGER.info("pretrain step %s / %s", human_format(step), human_format(num_steps))

        if eval_every and step % eval_every == 0:
            rates = per_stage_success_rate(
                agent,
                task_order=pretrain_tasks,
                num_episodes=min(eval_episodes, 10),
                seed=seed + step,
                stub=stub,
                deterministic=True,
            )
            mean_rate = float(np.mean([v for k, v in rates.items() if ":" not in k] or [0.0]))
            history.append({"step": step, "mean_success": mean_rate})
            if verbose:
                LOGGER.info(
                    "pretrain eval @ %s: mean success %.3f (%s)",
                    human_format(step),
                    mean_rate,
                    {k: round(v, 3) for k, v in rates.items() if ":" not in k},
                )

    agent.policy.eval()
    rates = per_stage_success_rate(
        agent,
        task_order=pretrain_tasks,
        num_episodes=eval_episodes,
        seed=seed + 999,
        stub=stub,
        deterministic=True,
    )

    ckpt_path = checkpoint or _cfg_get(cfg, "pretrain.checkpoint", None)
    if ckpt_path:
        ensure_dir(os.path.dirname(os.path.abspath(ckpt_path)) or ".")
        agent.save(ckpt_path)
        if verbose:
            LOGGER.info("saved pi_* to %s", ckpt_path)

    result = PretrainResult(
        tasks=pretrain_tasks,
        steps=num_steps,
        success_rates=rates,
        checkpoint=ckpt_path,
        history=history,
    )
    return agent, result


# --------------------------------------------------------------------------- #
# Retention construction
# --------------------------------------------------------------------------- #
def build_retention(
    cfg: Any,
    agent: SACAgent,
    *,
    method: Optional[str] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    teacher_agent: Optional[SACAgent] = None,
    pretrain_env: Optional[RoboticSequenceEnv] = None,
    stub: bool = False,
) -> Any:
    """Instantiate the requested actor-only retention mechanism.

    * ``ewc`` -- diagonal Fisher of the actor at ``theta_*`` (Wołczyk et al.
      2021 convention: squared log-likelihood gradients over 1000 sampled
      batches) with actor coefficient ``100`` (Table 3).
    * ``bc``  -- ``L_BC(theta) = E_{s ~ B_BC}[D_KL^s(pi_theta || pi_*)]`` with
      actor coefficient ``1`` and a ``10000``-sample buffer of pre-training
      states (Table 3).
    * ``em``  -- no auxiliary loss; 10% of the replay buffer is instead locked
      for ``pi_*`` transitions gathered on the prior tasks (Appendix C.3).
    """
    method = (method or _retention_method(cfg)).lower()
    if method in ("none", "vanilla", "", "false", "null"):
        return None

    seed = int(_cfg_get(cfg, "seed", 0) if seed is None else seed)
    batch_size = int(_cfg_get(cfg, "retention.bc.batch_size", 128))

    if method == "ewc":
        if EWC is None:
            raise RuntimeError("EWC unavailable (torch import failed)")
        actor_coef = float(_cfg_get(cfg, "retention.ewc.actor_coef", 100.0))
        fisher = None
        if FisherEstimator is not None and teacher_agent is not None:
            try:
                estimator = FisherEstimator(
                    teacher_agent.policy,
                    mode=str(_cfg_get(cfg, "retention.ewc.mode", "policy")),
                    num_batches=int(_cfg_get(cfg, "retention.ewc.num_batches", 1000)),
                )
                # Online sampling mode: roll out pi_* briefly to accumulate the
                # diagonal Fisher without needing an offline dataset.
                fisher = _fisher_from_policy_rollouts(
                    cfg,
                    teacher_agent,
                    estimator,
                    num_batches=int(_cfg_get(cfg, "retention.ewc.num_batches", 1000)),
                    batch_size=int(_cfg_get(cfg, "retention.ewc.batch_size", batch_size)),
                    seed=seed,
                    stub=stub,
                    pretrain_env=pretrain_env,
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Fisher estimation failed (%s); using uniform anchor.", exc)
        return EWC(
            agent.policy,
            fisher_diag=fisher,
            coef=actor_coef,
            normalize=bool(_cfg_get(cfg, "retention.ewc.normalize", False)),
        )

    if method == "bc":
        if BehavioralCloning is None:
            raise RuntimeError("BehavioralCloning unavailable (torch import failed)")
        actor_coef = float(_cfg_get(cfg, "retention.bc.actor_coef", 1.0))
        memory = int(_cfg_get(cfg, "retention.bc.memory_size", 10_000))
        buffer = None
        if build_bc_buffer is not None and teacher_agent is not None and pretrain_env is not None:
            states = _collect_pretrain_states(
                cfg, pretrain_env, teacher_agent, memory, seed=seed, stub=stub
            )
            if states:
                buffer = build_bc_buffer(
                    states,
                    teacher=teacher_agent.policy,
                    capacity=memory,
                    batch_size=batch_size,
                    device=device,
                    seed=seed,
                )
        bc = BehavioralCloning(
            agent.policy,
            teacher=teacher_agent.policy if teacher_agent is not None else None,
            buffer=buffer,
            coef=actor_coef,
            batch_size=batch_size,
            device=device,
        )
        decay = _cfg_get(cfg, "retention.bc.decay", None)
        if decay is not None:
            try:
                bc.configure({"decay": decay, "decay_type": _cfg_get(cfg, "retention.bc.decay_type", "exponential")})
            except Exception:  # pragma: no cover
                pass
        return bc

    if method in ("em", "episodic_memory", "memory"):
        if EpisodicMemory is None:
            raise RuntimeError("EpisodicMemory unavailable")
        em = EpisodicMemory(
            actor=agent.policy,
            capacity=int(_cfg_get(cfg, "retention.em.memory_size", _cfg_get(cfg, "sac.buffer_size", 100_000))),
            fraction=float(_cfg_get(cfg, "retention.em.fraction", 0.1)),
            batch_size=batch_size,
            env_name="robotic_sequence",
            device=device,
            seed=seed,
        )
        buffer = getattr(agent, "replay_buffer", None)
        if buffer is not None:
            try:
                if hasattr(buffer, "protected_fraction"):
                    buffer.protected_fraction = float(_cfg_get(cfg, "retention.em.fraction", 0.1))
            except Exception:  # pragma: no cover
                pass
            em.attach_buffer(buffer)
        # Pre-fill the protected region with pi_* transitions from prior stages.
        if teacher_agent is not None and pretrain_env is not None:
            try:
                _fill_em_with_pi_star(cfg, em, pretrain_env, teacher_agent, seed=seed, stub=stub)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("EM pre-fill failed: %s", exc)
        return em

    raise ValueError(f"unknown retention method: {method!r}")


def _fisher_from_policy_rollouts(
    cfg: Any,
    teacher_agent: SACAgent,
    estimator: Any,
    *,
    num_batches: int,
    batch_size: int,
    seed: int,
    stub: bool,
    pretrain_env: Optional[RoboticSequenceEnv],
) -> Any:
    """Accumulate the diagonal Fisher by rolling out ``pi_*`` on its own tasks.

    This mirrors the SAC/Continual World convention of Wołczyk et al. (2021):
    the Fisher is estimated from states visited by the pre-trained policy
    (rather than from an offline expert dataset as in NetHack).
    """
    import torch  # local import: only needed here

    env = pretrain_env or _build_env(cfg, resolve_task_order(cfg)[-2:], stub=stub, seed=seed)
    rng = make_generator(seed + 1234)
    obs = env.reset(seed=seed)
    collected = 0
    accum = 0
    try:
        estimator.reset()
    except Exception:  # pragma: no cover
        pass

    guard = 0
    max_guard = 100 * num_batches + 10_000
    while accum < num_batches and guard < max_guard:
        guard += 1
        batch_obs, batch_actions = [], []
        while len(batch_obs) < batch_size:
            with torch.no_grad():
                action = teacher_agent.select_action(
                    obs, stage_id=env.current_task.index, deterministic=False
                )
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            batch_obs.append(np.asarray(obs, dtype=np.float32))
            batch_actions.append(action)
            next_obs, _r, done, _info = env.step(action)
            obs = env.reset() if done else next_obs
        obs_t = torch.as_tensor(np.stack(batch_obs), dtype=torch.float32)
        act_t = torch.as_tensor(np.stack(batch_actions), dtype=torch.float32)
        estimator.accumulate(obs_t, act_t)
        accum += 1
    return estimator.diagonal()


def _collect_pretrain_states(
    cfg: Any,
    pretrain_env: RoboticSequenceEnv,
    teacher_agent: SACAgent,
    num_states: int,
    *,
    seed: int,
    stub: bool,
) -> List[Dict[str, Any]]:
    """Collect up to ``num_states`` states visited by ``pi_*`` for the BC buffer."""
    import torch  # noqa: F401  (ensures tensors are available)

    rng = make_generator(seed + 7)
    env = pretrain_env
    obs = env.reset(seed=seed)
    states: List[Dict[str, Any]] = []
    while len(states) < num_states:
        states.append({"obs": np.asarray(obs, dtype=np.float32)})
        action = teacher_agent.select_action(
            obs, stage_id=env.current_task.index, deterministic=False
        )
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_obs, _r, done, _info = env.step(action)
        obs = env.reset() if done else next_obs
        if stub and len(states) >= num_states:
            break
    return states


def _fill_em_with_pi_star(
    cfg: Any,
    em: Any,
    pretrain_env: RoboticSequenceEnv,
    teacher_agent: SACAgent,
    *,
    seed: int,
    stub: bool,
) -> None:
    """Roll out ``pi_*`` and insert the transitions into EM's protected region."""
    env = pretrain_env
    target = int(getattr(em, "protected_count", 0) or 0)
    capacity = int(_cfg_get(cfg, "retention.em.memory_size", 10_000))
    fraction = float(_cfg_get(cfg, "retention.em.fraction", 0.1))
    if target <= 0:
        target = int(round(fraction * capacity))
    obs = env.reset(seed=seed)
    added = 0
    guard = 0
    while added < target and guard < target * 5 + 1000:
        guard += 1
        action = teacher_agent.select_action(
            obs, stage_id=env.current_task.index, deterministic=False
        )
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_obs, reward, done, info = env.step(action)
        em.add(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": action,
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "done": bool(done),
                "stage_id": int(info.get("stage_id", env.current_task.index)),
            },
            protected=True,
        )
        added += 1
        obs = env.reset() if done else next_obs


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #
@dataclass
class FinetuneResult:
    """Record of one fine-tuning run."""

    method: str
    seed: int
    steps: int
    final_success_rates: Dict[str, float] = field(default_factory=dict)
    forgetting: Dict[str, float] = field(default_factory=dict)
    loglikelihood_trace: List[Dict[str, float]] = field(default_factory=list)
    eval_history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "steps": self.steps,
            "final_success_rates": self.final_success_rates,
            "forgetting": self.forgetting,
            "loglikelihood_trace": self.loglikelihood_trace,
            "eval_history": self.eval_history,
            "checkpoint": self.checkpoint,
        }


def evaluate_agent(
    agent: SACAgent,
    tasks: Sequence[str],
    *,
    num_episodes: int = 20,
    seed: int = 0,
    stub: bool = False,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Per-stage success rate + mean return for ``agent`` on ``tasks``."""
    rates = per_stage_success_rate(
        agent,
        task_order=tuple(tasks),
        num_episodes=num_episodes,
        seed=seed,
        stub=stub,
        deterministic=deterministic,
    )
    env = _build_env(None, tasks, stub=stub, seed=seed) if False else None  # placeholder
    return rates


def expert_log_likelihood(
    agent: SACAgent,
    teacher: SACAgent,
    task: str,
    *,
    num_episodes: int = 5,
    seed: int = 0,
    stub: bool = False,
) -> float:
    """Expert-action log-likelihood of ``agent`` on ``task`` (Figure 8).

    Rolls out the *expert* ``pi_*`` on ``task`` and averages the log-probability
    that the current policy ``pi_theta`` assigns to the expert's actions.
    """
    import torch

    env = _build_env(None, (task,), stub=stub, seed=seed)
    log_probs: List[float] = []
    for ep in range(num_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            with torch.no_grad():
                action = teacher.select_action(obs, stage_id=0, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            try:
                with torch.no_grad():
                    lp = agent.policy.log_prob(
                        torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0),
                        torch.as_tensor(action).unsqueeze(0),
                        stage_id=0,
                    )
                log_probs.append(float(torch.as_tensor(lp).reshape(-1)[0]))
            except Exception:  # pragma: no cover - policy signature mismatch
                pass
            obs, _r, done, _info = env.step(action)
    return float(np.mean(log_probs)) if log_probs else float("nan")


def finetune(
    cfg: Any,
    *,
    method: str = "none",
    stub: bool = False,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    num_steps: Optional[int] = None,
    pretrained: Optional[SACAgent] = None,
    teacher_agent: Optional[SACAgent] = None,
    pretrain_env: Optional[RoboticSequenceEnv] = None,
    checkpoint: Optional[str] = None,
    metric_logger: Optional[MetricLogger] = None,
) -> Tuple[SACAgent, FinetuneResult]:
    """Fine-tune ``pi_*`` on the full sequence with optional retention."""
    seed = int(_cfg_get(cfg, "seed", 0) if seed is None else seed)
    set_seed(seed)
    device = device or str(_cfg_get(cfg, "compute.device", "cpu"))
    num_steps = int(num_steps or _cfg_get(cfg, "finetune.num_steps", 250_000))
    method = (method or "none").lower()

    seq = resolve_task_order(cfg)
    env = _build_env(cfg, seq, stub=stub, seed=seed)
    obs_dim, action_dim, n_stages = env.observation_dim, env.action_dim, env.n_stages

    # Build the learner; if a pre-trained agent is supplied, copy its weights.
    agent = build_sac_agent(
        cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_stages=n_stages,
        device=device,
        retention=None,
        seed=seed,
    )
    if pretrained is not None:
        try:
            agent.load_state_dict(pretrained.state_dict(), load_optimizers=False)
            agent.policy.load_state_dict(pretrained.policy.state_dict())
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("could not copy pi_* weights (%s); initialising fresh", exc)

    teacher = teacher_agent if teacher_agent is not None else pretrained
    retention = build_retention(
        cfg,
        agent,
        method=method,
        device=device,
        seed=seed,
        teacher_agent=teacher,
        pretrain_env=pretrain_env,
        stub=stub,
    )
    if retention is not None:
        try:
            agent.set_retention(retention)
        except Exception:  # pragma: no cover
            agent.retention = retention

    # Per the paper, entropy is disabled when retention methods are enabled
    # (NetHack convention; also applied for the SAC ablation consistency).
    if retention is not None and hasattr(agent, "entropy_enabled"):
        try:
            agent.entropy_enabled = False
        except Exception:  # pragma: no cover
            pass

    batch_size = int(_cfg_get(cfg, "sac.batch_size", 128))
    update_every = int(_cfg_get(cfg, "sac.update_every", 1))
    updates_per_step = int(_cfg_get(cfg, "sac.updates_per_step", 1))
    warmup = int(_cfg_get(cfg, "sac.warmup_steps", 1000))
    eval_every = int(_cfg_get(cfg, "finetune.eval_every", 25_000))
    eval_episodes = int(_cfg_get(cfg, "finetune.eval_episodes", 20))
    save_every = int(_cfg_get(cfg, "finetune.save_every", 250_000))
    log_every = int(_cfg_get(cfg, "finetune.log_every", 10_000))
    ll_every = int(_cfg_get(cfg, "finetune.loglikelihood_every", 50_000))
    ll_task = str(_cfg_get(cfg, "finetune.loglikelihood_task", "push-wall"))

    rng = make_generator(seed)
    result = FinetuneResult(method=method, seed=seed, steps=num_steps)

    # Baseline success rates measured with pi_* before fine-tuning starts.
    baseline_rates: Dict[str, float] = {}
    if teacher is not None:
        baseline_rates = per_stage_success_rate(
            teacher, task_order=seq, num_episodes=min(eval_episodes, 20),
            seed=seed + 1, stub=stub, deterministic=True,
        )

    obs = env.reset(seed=seed)
    ep_return = 0.0
    for step in range(1, num_steps + 1):
        if step <= warmup:
            action = np.asarray(
                [rng.uniform(-1.0, 1.0) for _ in range(action_dim)], dtype=np.float32
            )
        else:
            action = agent.select_action(
                obs, stage_id=env.current_task.index, deterministic=False
            )
        action = np.asarray(action, dtype=np.float32).reshape(action_dim)

        next_obs, reward, done, info = env.step(action)
        buffer = getattr(agent, "replay_buffer", None)
        if buffer is not None:
            buffer.add(
                {
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "done": bool(done),
                    "stage_id": int(info.get("stage_id", env.current_task.index)),
                }
            )
        ep_return += float(reward)

        if step > warmup and step % max(update_every, 1) == 0 and len(buffer or []) >= batch_size:
            for _ in range(max(updates_per_step, 1)):
                stats = agent.update(batch_size=batch_size)
                if metric_logger is not None:
                    for k, v in (stats or {}).items():
                        metric_logger.add(f"train/{k}", v)
        if retention is not None:
            try:
                retention.step()
            except Exception:  # pragma: no cover
                pass

        obs = env.reset() if done else next_obs
        if done:
            ep_return = 0.0

        if step % log_every == 0:
            LOGGER.info(
                "[%s] step %s / %s", method, human_format(step), human_format(num_steps)
            )
            if metric_logger is not None:
                metric_logger.flush(step)

        if ll_every and step % ll_every == 0 and teacher is not None:
            ll = expert_log_likelihood(
                agent, teacher, ll_task, num_episodes=2, seed=seed + step, stub=stub
            )
            result.loglikelihood_trace.append({"step": step, "loglikelihood": ll})
            if metric_logger is not None:
                metric_logger.add("eval/expert_loglikelihood", ll)
            LOGGER.info("[%s] expert log-likelihood(%s) @ %s: %.4f",
                        method, ll_task, human_format(step), ll)

        if eval_every and step % eval_every == 0:
            rates = per_stage_success_rate(
                agent, task_order=seq, num_episodes=min(eval_episodes, 10),
                seed=seed + step, stub=stub, deterministic=True,
            )
            entry = {"step": step, **rates}
            result.eval_history.append(entry)
            if metric_logger is not None:
                for k, v in rates.items():
                    if ":" not in k:
                        metric_logger.add(f"eval/success_{k}", v)

    final_rates = per_stage_success_rate(
        agent, task_order=seq, num_episodes=eval_episodes,
        seed=seed + 4242, stub=stub, deterministic=True,
    )
    result.final_success_rates = final_rates

    # Forgetting: how much of the pre-trained (FAR) performance was lost.
    for task, base in (baseline_rates or {}).items():
        if ":" in task:
            continue
        result.forgetting[task] = float(base) - float(final_rates.get(task, 0.0))

    if checkpoint:
        ensure_dir(os.path.dirname(os.path.abspath(checkpoint)) or ".")
        try:
            agent.save(checkpoint)
            result.checkpoint = checkpoint
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("could not save checkpoint: %s", exc)

    try:
        env.close()
    except Exception:  # pragma: no cover
        pass
    return agent, result


# --------------------------------------------------------------------------- #
# From-scratch baseline
# --------------------------------------------------------------------------- #
def train_from_scratch(
    cfg: Any,
    *,
    stub: bool = False,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    num_steps: Optional[int] = None,
    metric_logger: Optional[MetricLogger] = None,
) -> Tuple[SACAgent, FinetuneResult]:
    """Vanilla SAC on the full sequence, starting from random weights."""
    return finetune(
        cfg,
        method="none",
        stub=stub,
        seed=seed,
        device=device,
        num_steps=num_steps,
        pretrained=None,
        teacher_agent=None,
        pretrain_env=None,
        metric_logger=metric_logger,
    )


# --------------------------------------------------------------------------- #
# Forward-transfer / prefix-length evaluation (Table 6)
# --------------------------------------------------------------------------- #
def forward_transfer_table(
    cfg: Any,
    agent: SACAgent,
    *,
    baseline_scores: Optional[Dict[int, float]] = None,
    prefix_lengths: Optional[Sequence[int]] = None,
    num_episodes: int = 20,
    stub: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute the forward-transfer metric of §F over prefix-task lengths.

    ``FT = (AUC - AUC^b) / (1 - AUC^b)`` where ``AUC`` is the area under the
    per-stage success-rate curve of the fine-tuned agent and ``AUC^b`` the same
    quantity for the from-scratch baseline.
    """
    seq = resolve_task_order(cfg)
    prefix_lengths = prefix_lengths or list(
        _cfg_get(cfg, "eval.prefix_lengths", [1, 2, 3, 4])
    )
    baseline_scores = baseline_scores or {}

    rows: List[Dict[str, Any]] = []
    for length in prefix_lengths:
        prefix = seq[: int(length)]
        rates = per_stage_success_rate(
            agent, task_order=prefix, num_episodes=num_episodes,
            seed=seed + int(length), stub=stub, deterministic=True,
        )
        per_stage = [v for k, v in rates.items() if ":" not in k]
        auc = float(np.mean(per_stage)) if per_stage else float("nan")
        base = baseline_scores.get(int(length))
        ft = (
            forward_transfer_metric(auc, base)
            if base is not None and auc == auc
            else float("nan")
        )
        rows.append(
            {
                "prefix_length": int(length),
                "tasks": list(prefix),
                "auc": auc,
                "baseline_auc": base,
                "forward_transfer": ft,
                "per_stage": {k: v for k, v in rates.items() if ":" not in k},
            }
        )
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# Full run orchestration
# --------------------------------------------------------------------------- #
def run_experiment(
    cfg: Any,
    *,
    stub: Optional[bool] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    write_metrics: bool = True,
) -> Dict[str, Any]:
    """Run pre-training + the requested retention method for one seed.

    Returns a JSON-serialisable summary with the per-stage success rates,
    forgetting scores, log-likelihood trace and forward-transfer table.
    """
    seed = int(_cfg_get(cfg, "seed", 0) if seed is None else seed)
    stub = bool(_cfg_get(cfg, "env.stub", False)) if stub is None else stub
    device = device or str(_cfg_get(cfg, "compute.device", "cpu"))
    method = _retention_method(cfg)

    set_seed(seed)
    out_dir = output_dir or os.path.join(
        str(_cfg_get(cfg, "output_dir", "runs")),
        "robotic_sequence",
        f"{method}_seed{seed}",
    )
    ensure_dir(out_dir)
    try:
        dump_config(cfg, os.path.join(out_dir, "config.yaml"))
    except Exception:  # pragma: no cover
        pass

    metric_logger = MetricLogger(out_dir, use_tensorboard=bool(_cfg_get(cfg, "logging.use_tensorboard", True))) if write_metrics else None

    # ---- (1) pre-train pi_* ------------------------------------------------ #
    pi_star, pretrain_result = pretrain_pistar(
        cfg, stub=stub, seed=seed, device=device
    )
    pretrain_env = _build_env(
        cfg, pretrain_result.tasks, stub=stub, seed=seed
    )

    # ---- (2) fine-tune ----------------------------------------------------- #
    ft_ckpt = os.path.join(out_dir, "finetuned.pt")
    agent, ft_result = finetune(
        cfg,
        method=method,
        stub=stub,
        seed=seed,
        device=device,
        pretrained=pi_star,
        teacher_agent=pi_star,
        pretrain_env=pretrain_env,
        checkpoint=ft_ckpt,
        metric_logger=metric_logger,
    )

    # ---- (3) from-scratch baseline (needed for forward transfer) ----------- #
    baseline_scores: Dict[int, float] = {}
    try:
        scratch_agent, scratch_result = train_from_scratch(
            cfg, stub=stub, seed=seed, device=device,
            num_steps=int(_cfg_get(cfg, "finetune.num_steps", 250_000)),
        )
        seq = resolve_task_order(cfg)
        prefix_lengths = list(_cfg_get(cfg, "eval.prefix_lengths", [1, 2, 3, 4]))
        for length in prefix_lengths:
            prefix = seq[: int(length)]
            rates = per_stage_success_rate(
                scratch_agent, task_order=prefix, num_episodes=10,
                seed=seed + int(length), stub=stub, deterministic=True,
            )
            per_stage = [v for k, v in rates.items() if ":" not in k]
            baseline_scores[int(length)] = (
                float(np.mean(per_stage)) if per_stage else float("nan")
            )
    except Exception as exc:  # pragma: no cover - baseline is best effort
        LOGGER.warning("from-scratch baseline failed: %s", exc)
        scratch_result = None

    # ---- (4) forward transfer --------------------------------------------- #
    ft_table = forward_transfer_table(
        cfg, agent, baseline_scores=baseline_scores, stub=stub, seed=seed
    )

    summary = {
        "name": "robotic_sequence",
        "method": method,
        "seed": seed,
        "pretrain": pretrain_result.as_dict(),
        "finetune": ft_result.as_dict(),
        "from_scratch": scratch_result.as_dict() if scratch_result is not None else None,
        "forward_transfer": ft_table,
        "output_dir": out_dir,
    }

    if write_metrics:
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        if metric_logger is not None:
            metric_logger.close()
    return summary


def aggregate_seeds(summaries: Sequence[Dict[str, Any]], confidence: float = 0.90) -> Dict[str, Any]:
    """Aggregate per-seed summaries into mean / CI metrics (20 seeds, 90% CI)."""
    per_stage: Dict[str, List[float]] = {}
    forgetting: Dict[str, List[float]] = {}
    for summary in summaries:
        ft = summary.get("finetune", {})
        for task, rate in (ft.get("final_success_rates") or {}).items():
            if ":" in task:
                continue
            per_stage.setdefault(task, []).append(float(rate))
        for task, delta in (ft.get("forgetting") or {}).items():
            forgetting.setdefault(task, []).append(float(delta))

    def _stats(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"mean": float("nan"), "half_width": float("nan"), "n": 0}
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        if arr.size < 2:
            return {"mean": mean, "half_width": float("nan"), "n": int(arr.size)}
        std = float(arr.std(ddof=1))
        try:
            from scipy import stats as _stats_mod  # type: ignore

            crit = float(_stats_mod.t.ppf(0.5 + confidence / 2.0, arr.size - 1))
        except Exception:  # pragma: no cover
            crit = 1.645  # normal approximation for 90% CI
        return {
            "mean": mean,
            "half_width": crit * std / float(np.sqrt(arr.size)),
            "std": std,
            "n": int(arr.size),
        }

    return {
        "num_seeds": len(summaries),
        "confidence": confidence,
        "success_rate": {task: _stats(vals) for task, vals in per_stage.items()},
        "forgetting": {task: _stats(vals) for task, vals in forgetting.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboticSequence SAC pre-train / fine-tune")
    parser.add_argument("--config", type=str, default="configs/robotic_sequence.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--method", type=str, default=None,
                        choices=["none", "ewc", "bc", "em", "vanilla"])
    parser.add_argument("--stub", action="store_true", help="force the stub Meta-World env")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--pretrain-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="run over multiple seeds and aggregate")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if os.path.exists(args.config):
        cfg = load_config(args.config, overrides=args.overrides)
    else:
        cfg = Config({})
        cfg["env"] = {"task_order": "main", "stub": True}
        cfg["sac"] = {}
        cfg["retention"] = {"method": args.method or "none"}
        cfg["finetune"] = {}
        cfg["eval"] = {"prefix_lengths": [1, 2, 3, 4]}
        cfg["seed"] = 0
        if args.overrides:
            apply_overrides(cfg, args.overrides)

    if args.method:
        cfg.setdefault("retention", Config({}))["method"] = args.method
    if args.stub:
        cfg.setdefault("env", Config({}))["stub"] = True
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.pretrain_steps is not None:
        cfg.setdefault("pretrain", Config({}))["num_steps"] = int(args.pretrain_steps)
    if args.num_steps is not None:
        cfg.setdefault("finetune", Config({}))["num_steps"] = int(args.num_steps)

    log_config(cfg, LOGGER)
    stub = bool(_cfg_get(cfg, "env.stub", False))

    if args.pretrain_only:
        _agent, result = pretrain_pistar(
            cfg,
            stub=stub,
            seed=args.seed,
            device=args.device,
            checkpoint=_cfg_get(cfg, "pretrain.checkpoint", None),
        )
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    seeds = args.seeds if args.seeds else [int(_cfg_get(cfg, "seed", 0))]
    summaries = []
    for seed in seeds:
        cfg_run = cfg.copy()
        cfg_run["seed"] = int(seed)
        summaries.append(
            run_experiment(
                cfg_run,
                stub=stub,
                seed=int(seed),
                device=args.device,
                output_dir=(
                    os.path.join(args.output_dir, f"seed{seed}") if args.output_dir else None
                ),
            )
        )

    output: Dict[str, Any] = {"runs": summaries}
    if len(seeds) > 1:
        output["aggregate"] = aggregate_seeds(
            summaries, confidence=float(_cfg_get(cfg, "eval.confidence", 0.90))
        )
    print(json.dumps(output.get("aggregate", output), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
