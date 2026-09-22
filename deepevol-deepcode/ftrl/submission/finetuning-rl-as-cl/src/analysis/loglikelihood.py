"""Expert-action log-likelihood analysis (Section 5, Figure 8).

For the RoboticSequence experiments the paper supplements the per-stage
success-rate curves with the *log-likelihood assigned by the fine-tuned policy
to trajectories collected using the expert policy* :math:`\\pi_*`, i.e. to the
state-action pairs :math:`(s, a^*)`, where :math:`a^* \\sim \\pi_*(s)`:

    "We supplement this analysis by studying the log-likelihoods assigned by the
    fine-tuned policy to trajectories collected using the expert policy, i.e.,
    the state-action pairs (s, a*), where a* ~ pi_*(s). This is visualized on
    Figure 8 where we show how the policy deteriorates in certain parts of the
    state space (projected to 2D using PCA) in the push-wall environment. After
    100K steps, the model assigns high probability to some of the correct actions
    on the part of the state space, but its overall success rate has already
    collapsed to 0. ... After the 500K steps, the likelihood values collapse on
    all expert trajectories. ... the log-likelihoods do not reach the original
    values, showing that the fine-tuned agent learned a different policy."

Concretely the quantity tracked is

.. math::

    \\bar{\\ell}(\\theta) \\;=\\; \\mathbb{E}_{(s, a^*) \\sim \\pi_*}
        \\big[ \\log \\pi_\\theta(a^* \\mid s) \\big],

evaluated on a *fixed* expert dataset (usually collected on ``push-wall``), which
is the first environment of the pre-training repertoire.  The plan fixes the
cadence to every ``50_000`` fine-tuning steps (``finetune.loglikelihood_every``).

This module therefore provides

* rollout collection with the frozen expert :math:`\\pi_*`
  (:func:`collect_expert_transitions`, :func:`collect_expert_dataset`),
* per-sample and aggregate log-likelihood evaluation of an arbitrary policy on
  such a dataset (:func:`expert_log_prob`, :func:`expert_log_likelihood`),
* an online tracker mirroring the JSON layout produced by
  ``src/robotic_sequence/train_robotic.py`` (:class:`LogLikelihoodTracker`) with
  the paper's qualitative claims as explicit metrics: the *collapse step* and the
  *recovery ratio* (the final value never reaches the pre-trained one),
* a Matplotlib rendering of Figure 8's top row,
* a small CLI for offline recomputation from checkpoints.

Everything torch-dependent is imported defensively so that the metric logic
(AUC-like reductions, aggregation, JSON IO) stays importable without PyTorch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


__all__ = [
    "DEFAULT_LL_EVERY",
    "DEFAULT_COLLAPSE_FRACTION",
    "DEFAULT_NUM_EXPERT_EPISODES",
    "ExpertDataset",
    "LogLikelihoodTracker",
    "expert_log_prob",
    "expert_log_likelihood",
    "loglikelihood_trace",
    "collect_expert_transitions",
    "collect_expert_dataset",
    "stack_expert_data",
    "collapse_step",
    "recovery_ratio",
    "aggregate_loglikelihood",
    "plot_loglikelihood",
    "main",
]

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

#: Log-likelihood is computed every 50K fine-tuning steps (config
#: ``finetune.loglikelihood_every``; Figure 8 has markers at 100K / 500K).
DEFAULT_LL_EVERY = 50_000

#: A policy is considered to have *forgotten* the expert solution once the mean
#: log-likelihood drops below this fraction of its pre-training value.  The paper
#: reports that the success rate has collapsed to 0 by 100K steps while the
#: likelihood collapses fully by 500K steps.
DEFAULT_COLLAPSE_FRACTION = 0.5

#: Number of expert rollouts gathered on the reference stage (push-wall).
DEFAULT_NUM_EXPERT_EPISODES = 5

_EPS = 1e-12


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError(
            "src.analysis.loglikelihood requires PyTorch for policy evaluation."
        )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _find_distribution(output: Any) -> Optional[Any]:
    """Locate a ``torch.distributions``-style object inside a model output."""
    if output is None:
        return None
    if hasattr(output, "log_prob") and not isinstance(output, (tuple, list, dict)):
        return output
    if isinstance(output, Mapping):
        for key in ("dist", "distribution", "action_dist", "policy_dist"):
            if key in output:
                found = _find_distribution(output[key])
                if found is not None:
                    return found
    if isinstance(output, (tuple, list)):
        for item in output:
            found = _find_distribution(item)
            if found is not None:
                return found
    for attr in ("dist", "distribution", "action_dist", "policy_dist"):
        cand = getattr(output, attr, None)
        if cand is not None and hasattr(cand, "log_prob"):
            return cand
    return None


def _find_logits(output: Any) -> Optional[Any]:
    if output is None:
        return None
    if isinstance(output, Mapping):
        for key in ("logits", "action_logits", "policy_logits"):
            if key in output:
                return output[key]
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if _HAS_TORCH and isinstance(first, torch.Tensor) and first.dim() >= 1:
            return first
    for attr in ("logits", "action_logits", "policy_logits"):
        cand = getattr(output, attr, None)
        if cand is not None:
            return cand
    return None


def _to_tensor(value: Any, device: Any = None) -> Any:
    if not _HAS_TORCH:  # pragma: no cover
        return value
    if isinstance(value, torch.Tensor):
        tensor = value
    elif _np is not None and isinstance(value, _np.ndarray):
        tensor = torch.as_tensor(value)
    elif isinstance(value, (list, tuple)):
        try:
            tensor = torch.as_tensor(_np.asarray(value))
        except Exception:
            return value
    else:
        return value
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _filter_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only the keyword arguments accepted by ``fn`` (duck-typed calls)."""
    try:
        import inspect

        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:  # pragma: no cover
        return dict(kwargs)


def _call_with_stage(fn: Callable[..., Any], obs: Any, stage_id: Any = None,
                     extra: Optional[Mapping[str, Any]] = None) -> Any:
    """Call a policy/forward function, transparently passing the stage id."""
    kwargs: Dict[str, Any] = dict(extra or {})
    if stage_id is not None:
        kwargs["stage_id"] = stage_id
    return fn(obs, **_filter_kwargs(fn, kwargs))


def _forward_policy(policy: Any, obs: Any, stage_id: Any = None,
                    actions: Any = None, extra: Optional[Mapping[str, Any]] = None) -> Any:
    """Obtain output (distribution / logits) from assorted policy interfaces."""
    # Dedicated evaluation helper, if the policy provides one.
    for name in ("log_prob", "log_probability", "action_log_prob"):
        fn = getattr(policy, name, None)
        if callable(fn) and actions is not None:
            try:
                return _call_with_stage(fn, obs, None, None) if False else fn(obs, actions)
            except Exception:
                pass
    for name in ("distribution", "dist", "get_distribution"):
        fn = getattr(policy, name, None)
        if callable(fn):
            try:
                return _call_with_stage(fn, obs, stage_id, extra)
            except Exception:
                try:
                    return fn(obs)
                except Exception:
                    pass
    if callable(policy):
        try:
            return _call_with_stage(policy, obs, stage_id, extra)
        except TypeError:
            pass
    fn = getattr(policy, "forward", None)
    if callable(fn):
        try:
            return _call_with_stage(fn, obs, stage_id, extra)
        except Exception:
            try:
                return fn(obs)
            except Exception:
                pass
    return None


def _scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu().item())
    if _np is not None and isinstance(value, _np.ndarray):
        return float(value.reshape(-1).mean())
    try:
        return float(value)
    except Exception:  # pragma: no cover
        return float("nan")


# ---------------------------------------------------------------------------
# Expert data collection
# ---------------------------------------------------------------------------

@dataclass
class ExpertDataset:
    """Fixed set of ``(s, a*)`` pairs collected with the expert policy.

    Attributes
    ----------
    observations:
        Object understood by the evaluated policy (tensor / dict / ndarray).
    actions:
        Expert actions (``a* ~ pi_*(s)``), same length as ``observations``.
    stage_ids:
        Optional per-sample stage id (RoboticSequence per-stage heads).
    stage:
        Human readable stage name (``"push-wall"`` for Figure 8).
    rewards / dones / returns / steps_per_run:
        Diagnostics kept for the PCA / density visualisations.
    """

    observations: Any
    actions: Any
    stage_ids: Any = None
    stage: Optional[str] = None
    rewards: Any = None
    dones: Any = None
    returns: Optional[List[float]] = None
    steps_per_run: Optional[List[int]] = None
    is_far: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        obs = self.observations
        if isinstance(obs, Mapping):
            for value in obs.values():
                if hasattr(value, "__len__"):
                    return len(value)  # type: ignore[arg-type]
            return 0
        try:
            return len(obs)  # type: ignore[arg-type]
        except TypeError:
            return int(_scalar(obs).__bool__()) if obs is not None else 0

    @property
    def n_samples(self) -> int:
        return len(self)

    def to_dict(self, include_observations: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "stage": self.stage,
            "is_far": self.is_far,
            "n_samples": len(self),
            "returns": list(self.returns) if self.returns is not None else None,
            "steps_per_run": list(self.steps_per_run) if self.steps_per_run else None,
            "metadata": dict(self.metadata),
        }
        if include_observations:
            out["observations"] = _serialisable(self.observations)
            out["actions"] = _serialisable(self.actions)
            out["stage_ids"] = _serialisable(self.stage_ids)
        return out

    def save(self, path: str) -> str:
        payload = {
            "stage": self.stage,
            "is_far": self.is_far,
            "returns": list(self.returns) if self.returns is not None else None,
            "steps_per_run": list(self.steps_per_run) if self.steps_per_run else None,
            "metadata": dict(self.metadata),
            "observations": _serialisable(self.observations),
            "actions": _serialisable(self.actions),
            "stage_ids": _serialisable(self.stage_ids),
        }
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle)
        return path

    @classmethod
    def load(cls, path: str) -> "ExpertDataset":
        with open(path) as handle:
            payload = json.load(handle)
        return cls(
            observations=_from_serialisable(payload.get("observations")),
            actions=_from_serialisable(payload.get("actions")),
            stage_ids=_from_serialisable(payload.get("stage_ids")),
            stage=payload.get("stage"),
            returns=payload.get("returns"),
            steps_per_run=payload.get("steps_per_run"),
            is_far=bool(payload.get("is_far", False)),
            metadata=payload.get("metadata") or {},
        )


def _serialisable(value: Any) -> Any:
    if value is None:
        return None
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if _np is not None and isinstance(value, _np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {k: _serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(v) for v in value]
    return value


def _from_serialisable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {k: _from_serialisable(v) for k, v in value.items()}
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            if _HAS_TORCH:
                try:
                    return torch.as_tensor(value)
                except Exception:
                    return value
            if _np is not None:
                return _np.asarray(value)
        return value
    return value


def _select_action(policy: Any, obs: Any, stage_id: Any = None,
                   deterministic: bool = True) -> Any:
    """Robust action selection for duck-typed policies."""
    for name in ("act", "select_action", "sample_action", "get_action"):
        fn = getattr(policy, name, None)
        if callable(fn):
            kwargs: Dict[str, Any] = {}
            if stage_id is not None:
                kwargs["stage_id"] = stage_id
            kwargs["deterministic"] = deterministic
            try:
                out = fn(obs, **_filter_kwargs(fn, kwargs))
            except Exception:
                try:
                    out = fn(obs)
                except Exception:
                    continue
            return _extract_action(out)
    if callable(policy):
        try:
            kwargs = {"stage_id": stage_id, "deterministic": deterministic} if stage_id is not None else {
                "deterministic": deterministic}
            out = policy(obs, **_filter_kwargs(policy, kwargs))
        except Exception:
            try:
                out = policy(obs)
            except Exception:
                return None
        return _extract_action(out)
    fn = getattr(policy, "forward", None)
    if callable(fn):
        try:
            out = fn(obs)
        except Exception:
            return None
        return _extract_action(out)
    return None


def _extract_action(out: Any) -> Any:
    if out is None:
        return None
    if isinstance(out, Mapping):
        for key in ("action", "actions", "a"):
            if key in out:
                return out[key]
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    return out


def collect_expert_transitions(
    env: Any,
    teacher: Any,
    task: Optional[str] = None,
    num_episodes: int = DEFAULT_NUM_EXPERT_EPISODES,
    seed: Optional[int] = 0,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    stage_id: Optional[int] = None,
    stub: bool = False,
    initial_states: Optional[Sequence[Any]] = None,
) -> ExpertDataset:
    """Roll out the frozen expert :math:`\\pi_*` and record ``(s, a*)`` pairs.

    Rollouts happen on ``task`` (``push-wall`` for Figure 8) with the expert
    acting greedily, mirroring "trajectories collected using the expert policy".

    Parameters
    ----------
    env:
        A ``RoboticSequenceEnv`` (or a plain stage environment).  When a stage
        name is given and the environment is a sequence wrapper, the stage is set
        through ``reset(stage_id=...)``.
    teacher:
        Frozen :math:`\\pi_*` (any duck-typed policy).
    task:
        Stage name used for the rollouts (metadata only when ``env`` is a stage
        environment).
    num_episodes:
        Number of expert episodes (defaults: 5).
    max_steps:
        Per-episode step cap (defaults to the environment time limit ``T=200``).
    stage_id:
        Explicit stage index for per-stage policy heads.
    """
    observations: List[Any] = []
    actions: List[Any] = []
    rewards: List[float] = []
    dones: List[bool] = []
    returns: List[float] = []
    steps_per_run: List[int] = []
    stage_ids: List[int] = []

    resolved_stage = stage_id
    if resolved_stage is None and isinstance(task, int):
        resolved_stage = int(task)
        task = None
    if resolved_stage is None and isinstance(task, str) and task:
        resolved_stage = _default_stage_id(task)

    for episode in range(int(num_episodes)):
        ep_seed = None if seed is None else int(seed) + episode
        try:
            reset_kwargs: Dict[str, Any] = {}
            if ep_seed is not None:
                reset_kwargs["seed"] = ep_seed
            if resolved_stage is not None:
                reset_kwargs["stage_id"] = resolved_stage
            out = env.reset(**_filter_kwargs(env.reset, reset_kwargs))
        except TypeError:
            out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        stage_now = _info_stage(info, resolved_stage)
        done = False
        episode_return = 0.0
        steps = 0
        while not done:
            action = _select_action(teacher, obs, stage_now, deterministic=deterministic)
            if action is None:
                break
            observations.append(obs)
            actions.append(action)
            stage_ids.append(int(stage_now) if stage_now is not None else 0)
            out = env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated) or bool(truncated)
            else:
                obs, reward, done, info = out
                done = bool(done)
            rewards.append(float(reward))
            dones.append(bool(done))
            episode_return += float(reward)
            steps += 1
            if max_steps is not None and steps >= int(max_steps):
                break
            stage_new = _info_stage(info, stage_now)
            if stage_new is not None:
                stage_now = stage_new
        returns.append(episode_return)
        steps_per_run.append(steps)

    obs_stack = _stack_values(observations)
    act_stack = _stack_values(actions)
    return ExpertDataset(
        observations=obs_stack,
        actions=act_stack,
        stage_ids=stage_ids or None,
        stage=str(task) if task is not None else None,
        rewards=rewards,
        dones=dones,
        returns=returns,
        steps_per_run=steps_per_run,
        is_far=_is_far_task(task),
        metadata={"num_episodes": int(num_episodes), "seed": seed, "stub": bool(stub)},
    )


def _info_stage(info: Any, fallback: Optional[int]) -> Optional[int]:
    if isinstance(info, Mapping):
        for key in ("stage_id", "stage", "task_id"):
            if key in info and info[key] is not None:
                try:
                    return int(info[key])
                except Exception:
                    continue
    return fallback


def _default_stage_id(task: Optional[str]) -> Optional[int]:
    if not isinstance(task, str):
        return None
    try:
        from src.robotic_sequence.env import ROBOTIC_SEQUENCE_TASKS, strip_version

        name = strip_version(task)
        if name in ROBOTIC_SEQUENCE_TASKS:
            return ROBOTIC_SEQUENCE_TASKS.index(name)
    except Exception:
        pass
    return None


def _is_far_task(task: Optional[str]) -> bool:
    if not isinstance(task, str):
        return False
    try:
        from src.robotic_sequence.env import FAR_TASKS, strip_version

        return strip_version(task) in FAR_TASKS
    except Exception:
        return task in ("peg-unplug-side", "push-wall")


def _stack_values(values: Sequence[Any]) -> Any:
    if not values:
        return []
    first = values[0]
    if _HAS_TORCH and isinstance(first, torch.Tensor):
        try:
            return torch.stack([v.detach().cpu() for v in values], dim=0)
        except Exception:
            return list(values)
    if _np is not None and isinstance(first, _np.ndarray):
        try:
            return _np.stack([_np.asarray(v) for v in values], axis=0)
        except Exception:
            return list(values)
    if isinstance(first, Mapping):
        keys = list(first.keys())
        return {k: _stack_values([v[k] for v in values]) for k in keys}
    return list(values)


def collect_expert_dataset(*args: Any, **kwargs: Any) -> ExpertDataset:
    """Alias of :func:`collect_expert_transitions` (Figure 8 dataset)."""
    return collect_expert_transitions(*args, **kwargs)


def stack_expert_data(data: ExpertDataset, batch_size: Optional[int] = None) -> List[Any]:
    """Yield mini-batches of observations (with imitated actions) from a dataset."""
    n = len(data)
    size = int(batch_size) if batch_size else n
    batches: List[Any] = []
    for start in range(0, n, max(1, size)):
        stop = min(n, start + max(1, size))
        batches.append(_slice_values(data.observations, start, stop))
    return batches


def _slice_values(values: Any, start: int, stop: int) -> Any:
    if isinstance(values, Mapping):
        return {k: _slice_values(v, start, stop) for k, v in values.items()}
    try:
        return values[start:stop]
    except Exception:
        return values


# ---------------------------------------------------------------------------
# Log-likelihood evaluation
# ---------------------------------------------------------------------------

def _distribution_from(policy: Any, obs: Any, stage_id: Any = None,
                       extra: Optional[Mapping[str, Any]] = None) -> Any:
    out = _forward_policy(policy, obs, stage_id, None, extra)
    dist = _find_distribution(out)
    if dist is not None:
        return dist
    logits = _find_logits(out)
    if logits is not None and _HAS_TORCH:
        return torch.distributions.Categorical(logits=logits)
    return None


def expert_log_prob(
    policy: Any,
    observations: Any,
    actions: Any,
    stage_id: Any = None,
    step_ids: Any = None,
    batch_size: Optional[int] = None,
    device: Any = None,
    forward_fn: Optional[Callable[..., Any]] = None,
    extra_forward_kwargs: Optional[Mapping[str, Any]] = None,
    reduction: Optional[str] = "mean",
) -> Any:
    """Per-sample ``log pi_theta(a* | s)`` for expert actions.

    Supports the SAC squashed-Gaussian policy (``SACPolicy``), categorical
    policies via ``logits`` and any object exposing a ``distribution``/``forward``
    returning a ``torch.distributions`` instance.  ``stage_id`` may be a scalar
    (RoboticSequence stage index) or a per-sample sequence; ``step_ids`` is an
    accepted alias for per-sample stage identifiers.

    Returns a tensor of per-sample values, or their mean when
    ``reduction == "mean"``.
    """
    _require_torch()
    n = _length_of(observations)
    if n == 0:
        zero = torch.zeros((), dtype=torch.float32)
        return zero

    ids = step_ids if step_ids is not None else stage_id
    id_list = _as_index_list(ids, n)

    outputs: List[Any] = []
    chunk = int(batch_size) if batch_size else n
    for start in range(0, n, max(1, chunk)):
        stop = min(n, start + max(1, chunk))
        obs_b = _slice_obs(observations, start, stop, device)
        act_b = _to_tensor(_slice_raw(actions, start, stop), device)
        sid = None
        if id_list is not None:
            sid_vals = id_list[start:stop]
            sid = sid_vals[0] if len(set(sid_vals)) == 1 else _to_tensor(sid_vals, device)
        dist = None
        if forward_fn is not None:
            out = forward_fn(policy, obs_b, sid)
            dist = _find_distribution(out)
        if dist is None:
            dist = _distribution_from(policy, obs_b, sid, extra_forward_kwargs)
        if dist is None:
            raise RuntimeError(
                "Could not obtain a distribution from the policy. Provide "
                "`forward_fn` or expose `distribution`/`forward` returning a "
                "torch.distributions object."
            )
        lp = dist.log_prob(act_b)
        if isinstance(lp, Mapping):
            lp = next(iter(lp.values()))
        outputs.append(lp.reshape(-1))

    values = torch.cat(outputs, dim=0)
    if reduction is None:
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unknown reduction: {reduction!r}")


def expert_log_likelihood(policy: Any, dataset: Any, **kwargs: Any) -> float:
    """Mean expert-action log-likelihood of ``policy`` on an expert dataset.

    Accepts either an :class:`ExpertDataset` or the raw ``(observations,
    actions)`` pair.
    """
    if isinstance(dataset, ExpertDataset):
        obs, actions = dataset.observations, dataset.actions
        kwargs.setdefault("stage_id", dataset.stage_ids)
    else:  # pragma: no cover - convenience overload
        obs, actions = dataset
    return _scalar(expert_log_prob(policy, obs, actions, **kwargs))


def loglikelihood_trace(policy: Any, dataset: ExpertDataset,
                        batch_size: Optional[int] = None) -> Any:
    """Per-sample log-likelihoods, used by the PCA colour-coding of Figure 8."""
    return expert_log_prob(
        policy,
        dataset.observations,
        dataset.actions,
        stage_id=dataset.stage_ids,
        batch_size=batch_size,
        reduction=None,
    )


def _length_of(obs: Any) -> int:
    if isinstance(obs, Mapping):
        for value in obs.values():
            if hasattr(value, "__len__"):
                return len(value)  # type: ignore[arg-type]
        return 0
    try:
        return len(obs)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _as_index_list(ids: Any, n: int) -> Optional[List[int]]:
    if ids is None:
        return None
    if isinstance(ids, (int,)) and not isinstance(ids, bool):
        return [int(ids)] * n
    if _HAS_TORCH and isinstance(ids, torch.Tensor):
        flat = ids.detach().cpu().reshape(-1).tolist()
        if len(flat) == 1:
            flat = flat * n
        return [int(v) for v in flat]
    if _np is not None and isinstance(ids, _np.ndarray):
        flat = ids.reshape(-1).tolist()
        if len(flat) == 1:
            flat = flat * n
        return [int(v) for v in flat]
    if isinstance(ids, (list, tuple)):
        flat = list(ids)
        if len(flat) == 1:
            flat = flat * n
        return [int(v) for v in flat]
    return [int(ids)] * n


def _slice_raw(values: Any, start: int, stop: int) -> Any:
    if isinstance(values, Mapping):
        return {k: _slice_raw(v, start, stop) for k, v in values.items()}
    return values[start:stop]


def _slice_obs(obs: Any, start: int, stop: int, device: Any = None) -> Any:
    if isinstance(obs, Mapping):
        return {k: _move(_slice_obs(v, start, stop, None), device) for k, v in obs.items()}
    sliced = obs[start:stop]
    return _move(sliced, device)


def _move(value: Any, device: Any) -> Any:
    if device is None or not _HAS_TORCH:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: _move(v, device) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@dataclass
class LogLikelihoodRecord:
    step: int
    loglikelihood: float
    per_stage: Dict[str, float] = field(default_factory=dict)
    stage: Optional[str] = None
    n_samples: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "loglikelihood": float(self.loglikelihood),
            "per_stage": {k: float(v) for k, v in self.per_stage.items()},
            "stage": self.stage,
            "n_samples": int(self.n_samples),
        }


class LogLikelihoodTracker:
    """Tracks ``E[log pi_theta(a*|s)]`` on a fixed expert dataset (Figure 8).

    Parameters
    ----------
    dataset:
        Expert ``(s, a*)`` pairs collected with :math:`\\pi_*`.
    batch_size:
        Optional mini-batch size for the evaluation forward passes.
    device:
        Torch device for the evaluation.
    forward_fn:
        Optional callable ``(policy, obs, stage_id) -> output`` customising how
        the policy distribution is obtained (needed for exotic models).
    extra_forward_kwargs:
        Extra keyword arguments forwarded to the policy.

    Notes
    -----
    The tracker also stores the *pre-training* value :math:`\\bar\\ell(\\theta_*)`
    (either supplied through ``reference_value`` or measured lazily on the first
    ``record`` call when the evaluated model already is :math:`\\pi_*`), which is
    required to report the paper's qualitative claims: the collapse step and the
    fact that "the log-likelihoods do not reach the original values".
    """

    def __init__(
        self,
        dataset: ExpertDataset,
        batch_size: Optional[int] = None,
        device: Any = None,
        forward_fn: Optional[Callable[..., Any]] = None,
        extra_forward_kwargs: Optional[Mapping[str, Any]] = None,
        reference_value: Optional[float] = None,
        every: Optional[int] = DEFAULT_LL_EVERY,
        name: str = "loglikelihood",
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.forward_fn = forward_fn
        self.extra_forward_kwargs = dict(extra_forward_kwargs or {})
        self.reference_value = float(reference_value) if reference_value is not None else None
        self.every = every
        self.name = name
        self.records: List[LogLikelihoodRecord] = []
        self._last_step: Optional[int] = None
        self._next_due: Optional[int] = None if every is None else 0

    # -- evaluation -------------------------------------------------------
    def measure(self, policy: Any, step: Optional[int] = None) -> float:
        values = loglikelihood_trace(
            policy, self.dataset, self.batch_size
        ) if self.forward_fn is None else expert_log_prob(
            policy,
            self.dataset.observations,
            self.dataset.actions,
            stage_id=self.dataset.stage_ids,
            batch_size=self.batch_size,
            device=self.device,
            forward_fn=self.forward_fn,
            extra_forward_kwargs=self.extra_forward_kwargs,
            reduction=None,
        )
        return _scalar(values)

    def record(self, step: int, policy: Any,
               loglikelihood: Optional[float] = None) -> Optional[LogLikelihoodRecord]:
        """Record the metric at ``step`` (skipped when not due, if ``every`` set)."""
        step = int(step)
        if loglikelihood is None:
            if self.every is not None and self._next_due is not None:
                if step < self._next_due:
                    return None
            loglikelihood = expert_log_likelihood(
                policy,
                self.dataset,
                batch_size=self.batch_size,
                device=self.device,
                forward_fn=self.forward_fn,
                extra_forward_kwargs=self.extra_forward_kwargs,
            )
            if self.every is not None:
                self._next_due = step + int(self.every)
        record = LogLikelihoodRecord(
            step=step,
            loglikelihood=float(loglikelihood),
            stage=self.dataset.stage,
            n_samples=len(self.dataset),
        )
        self.records.append(record)
        self._last_step = step
        return record

    def record_step(self, step: int, policy: Any) -> Optional[LogLikelihoodRecord]:
        return self.record(step, policy)

    def set_reference(self, value: float) -> float:
        self.reference_value = float(value)
        return self.reference_value

    def measure_reference(self, teacher: Any) -> float:
        return self.set_reference(self.measure(teacher))

    # -- accessors --------------------------------------------------------
    @property
    def steps(self) -> List[int]:
        return [r.step for r in self.records]

    @property
    def values(self) -> List[float]:
        return [r.loglikelihood for r in self.records]

    @property
    def last_step(self) -> Optional[int]:
        return self._last_step

    @property
    def last_value(self) -> Optional[float]:
        return self.records[-1].loglikelihood if self.records else None

    def curve(self, relative: bool = False) -> Tuple[List[int], List[float]]:
        """``(steps, values)``; ``relative=True`` divides by the reference value."""
        values = list(self.values)
        if relative:
            ref = self.reference_value
            if ref is None or abs(ref) < _EPS:
                values = [float("nan")] * len(values)
            else:
                values = [v / ref for v in values]
        return list(self.steps), values

    def minimum(self) -> Optional[LogLikelihoodRecord]:
        return min(self.records, key=lambda r: r.loglikelihood) if self.records else None

    def maximum(self) -> Optional[LogLikelihoodRecord]:
        return max(self.records, key=lambda r: r.loglikelihood) if self.records else None

    # -- paper metrics ----------------------------------------------------
    def collapse_step(self, fraction: float = DEFAULT_COLLAPSE_FRACTION,
                      reference: Optional[float] = None) -> Optional[int]:
        """First step at which the value drops below ``fraction * reference``."""
        ref = reference if reference is not None else self.reference_value
        for record in self.records:
            if ref is None:
                # Relative to the first measurement when no reference is known.
                ref = self.records[0].loglikelihood
            if record.loglikelihood <= fraction * float(ref):
                return record.step
        return None

    def recovered(self, fraction: float = 0.9,
                  reference: Optional[float] = None) -> bool:
        """Whether the final value is back within ``fraction`` of the reference."""
        ref = reference if reference is not None else self.reference_value
        if ref is None or not self.records:
            return False
        return self.records[-1].loglikelihood >= fraction * float(ref)

    def recovery_ratio(
        self,
        reference: Optional[float] = None,
        window: int = 1,
    ) -> float:
        """Mean of the last ``window`` values over the reference value.

        The paper observes this ratio stays below 1 even after relearning
        ("the log-likelihoods do not reach the original values").
        """
        ref = reference if reference is not None else self.reference_value
        if ref is None or abs(float(ref)) < _EPS or not self.records:
            return float("nan")
        tail = self.records[-max(1, int(window)):]
        return float(sum(r.loglikelihood for r in tail) / len(tail) / float(ref))

    def recovered_fraction(self, reference: Optional[float] = None) -> float:
        """Fraction of the reference value reached at the end of training."""
        return self.recovery_ratio(reference=reference)

    def summary(self) -> Dict[str, Any]:
        ref = self.reference_value
        first = self.records[0].loglikelihood if self.records else None
        return {
            "name": self.name,
            "stage": self.dataset.stage,
            "n_records": len(self.records),
            "reference_value": ref,
            "initial_value": first,
            "final_value": self.last_value,
            "minimum": self.minimum().as_dict() if self.minimum() else None,
            "collapse_step": self.collapse_step(),
            "recovery_ratio": self.recovery_ratio(),
            "recovered": self.recovered(),
        }

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "records": [r.as_dict() for r in self.records],
            "reference_value": self.reference_value,
            "every": self.every,
            "stage": self.dataset.stage,
            "n_samples": len(self.dataset),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.records = [
            LogLikelihoodRecord(
                step=int(r.get("step", 0)),
                loglikelihood=float(r.get("loglikelihood", float("nan"))),
                per_stage=dict(r.get("per_stage") or {}),
                stage=r.get("stage"),
                n_samples=int(r.get("n_samples", 0)),
            )
            for r in state.get("records", [])
        ]
        ref = state.get("reference_value", None)
        self.reference_value = None if ref is None else float(ref)
        if state.get("every", None) is not None:
            self.every = int(state["every"])
        if self.records:
            self._last_step = self.records[-1].step

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle)
        return path

    def to_dict(self) -> Dict[str, Any]:
        data = self.state_dict()
        data["summary"] = self.summary()
        data["curve"] = {"steps": self.steps, "values": self.values}
        return data


# ---------------------------------------------------------------------------
# Free functions mirroring the tracker metrics
# ---------------------------------------------------------------------------

def collapse_step(steps: Sequence[int], values: Sequence[float],
                  fraction: float = DEFAULT_COLLAPSE_FRACTION,
                  reference: Optional[float] = None) -> Optional[int]:
    """First step whose log-likelihood falls below ``fraction * reference``."""
    if not values:
        return None
    ref = float(reference) if reference is not None else float(values[0])
    for step, value in zip(steps, values):
        if float(value) <= fraction * ref:
            return int(step)
    return None


def recovery_ratio(values: Sequence[float], reference: Optional[float] = None,
                   window: int = 1) -> float:
    """Final (or windowed-tail) value divided by the reference value."""
    if not values:
        return float("nan")
    ref = float(reference) if reference is not None else float(values[0])
    if abs(ref) < _EPS:
        return float("nan")
    tail = list(values)[-max(1, int(window)):]
    return float(sum(tail) / len(tail) / ref)


def aggregate_loglikelihood(records: Sequence[Any], key: str = "loglikelihood",
                            confidence: float = 0.90) -> Dict[str, Any]:
    """Aggregate per-seed records into the reported mean / 90% CI.

    ``records`` may be a sequence of floats, of ``LogLikelihoodRecord``/mappings
    with a ``step``/``key`` structure, or of ``(step, value)`` pairs; values are
    reduced per step so that multi-step curves are averaged across seeds.
    """
    per_step: Dict[int, List[float]] = {}
    flat: List[float] = []
    for item in records:
        if isinstance(item, (int, float)):
            flat.append(float(item))
            continue
        if isinstance(item, Mapping):
            step = int(item.get("step", len(per_step)))
            value = item.get(key, item.get("value"))
        elif isinstance(item, LogLikelihoodRecord):
            step, value = item.step, item.loglikelihood
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            step, value = int(item[0]), float(item[1])
        else:  # pragma: no cover
            step, value = len(per_step), float(_scalar(item))  # type: ignore[arg-type]
        if value is None:
            continue
        per_step.setdefault(int(step), []).append(float(value))
        flat.append(float(value))
    stats = summarize(flat, confidence=confidence) if flat else {
        "mean": float("nan"), "half_width": float("nan"), "n": 0, "std": float("nan")}
    per_step_stats = {
        step: summarize(vals, confidence=confidence) for step, vals in sorted(per_step.items())
    }
    stats["per_step"] = per_step_stats
    return stats


def summarize(values: Sequence[float], confidence: float = 0.90) -> Dict[str, float]:
    """Mean and normal-approximation confidence interval half-width."""
    vals = [float(v) for v in values if not math.isnan(float(v))]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "half_width": float("nan"), "n": 0, "std": float("nan")}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "half_width": 0.0, "n": 1, "std": 0.0}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    half_width = _z_for(confidence) * std / math.sqrt(n)
    return {"mean": mean, "half_width": half_width, "n": n, "std": std}


def _z_for(confidence: float) -> float:
    """Two-sided normal quantile (0.90 -> 1.6449, 0.95 -> 1.9600)."""
    table = {0.5: 0.6745, 0.68: 0.9945, 0.8: 1.2816, 0.9: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    for level, z in table.items():
        if abs(confidence - level) < 1e-6:
            return z
    if _np is not None:
        try:
            from math import erf, sqrt

            # inverse of the normal CDF via bisection on the closed-form CDF
            target = 0.5 + confidence / 2.0
            lo, hi = -10.0, 10.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                cdf = 0.5 * (1.0 + erf(mid / sqrt(2.0)))
                if cdf < target:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        except Exception:  # pragma: no cover
            pass
    return 1.6449


# ---------------------------------------------------------------------------
# Plotting (Figure 8, top row)
# ---------------------------------------------------------------------------

def plot_loglikelihood(tracker: Any, path: Optional[str] = None,
                       title: Optional[str] = None, show: bool = False,
                       label: Optional[str] = None, ax: Any = None) -> Any:
    """Plot the expert-action log-likelihood curve (Figure 8, top row)."""
    try:  # pragma: no cover - optional dependency
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plot_loglikelihood") from exc

    if isinstance(tracker, Mapping):
        steps = list(tracker.get("steps", []))
        values = list(tracker.get("values", []))
        summary = tracker.get("summary", {})
        stage = summary.get("stage", "push-wall")
    else:
        steps, values = tracker.curve()
        summary = tracker.summary()
        stage = tracker.dataset.stage

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, values, marker="o", ms=3, label=label or "fine-tuning")
    ref = summary.get("reference_value")
    if ref is not None:
        ax.axhline(ref, ls="--", lw=1, color="tab:green", label=r"$\pi_*$")
    collapse = summary.get("collapse_step")
    if collapse is not None:
        ax.axvline(collapse, ls=":", lw=1, color="tab:red", label="collapse")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(r"$\mathbb{E}_{s,a^*\sim\pi_*}[\log \pi_\theta(a^*|s)]$")
    ax.set_title(title or f"Expert log-likelihood ({stage or 'stage'})")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
    if path:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    if show:  # pragma: no cover
        plt.show()
    return ax.figure


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expert-action log-likelihood analysis (Section 5 / Figure 8). "
            "Evaluates E[log pi_theta(a*|s)] of a RoboticSequence checkpoint on a "
            "dataset collected with the pre-trained expert pi_*."
        )
    )
    parser.add_argument("--config", type=str, default=None,
                        help="configs/robotic_sequence.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None,
                        help="config overrides, e.g. seed=1")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="fine-tuned agent checkpoint to evaluate")
    parser.add_argument("--teacher", type=str, default=None,
                        help="pre-trained pi_* checkpoint (defines the dataset)")
    parser.add_argument("--stage", type=str, default="push-wall",
                        help="stage used for the expert dataset")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EXPERT_EPISODES)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--entropy-disabled", action="store_true",
                        help="build agents with automatic entropy tuning off")
    parser.add_argument("--stub", action="store_true",
                        help="use the CPU-only dummy stage environment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path for the measured value")
    parser.add_argument("--plot", type=str, default=None,
                        help="matplotlib output path for the curve")
    parser.add_argument("--dataset", type=str, default=None,
                        help="save/load the expert dataset JSON here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = None
    if args.config:
        try:
            from src.common.config import apply_overrides, load_config

            cfg = load_config(args.config, overrides=args.overrides)
        except Exception as exc:  # pragma: no cover
            print(f"[loglikelihood] could not load config: {exc}")
            cfg = None

    # -- build the environment + expert dataset --------------------------
    env = None
    dataset: Optional[ExpertDataset] = None
    try:
        from src.robotic_sequence.env import RoboticSequenceEnv

        env = RoboticSequenceEnv(stub=bool(args.stub), seed=args.seed)
    except Exception as exc:  # pragma: no cover
        print(f"[loglikelihood] environment unavailable: {exc}")

    teacher = None
    if args.teacher:
        try:
            from src.robotic_sequence.sac import build_sac_agent

            teacher = _load_agent(build_sac_agent, args.teacher, env, cfg, args)
        except Exception as exc:  # pragma: no cover
            print(f"[loglikelihood] could not load teacher: {exc}")

    if args.dataset and os.path.exists(args.dataset):
        dataset = ExpertDataset.load(args.dataset)
        print(f"[loglikelihood] loaded {len(dataset)} expert samples from {args.dataset}")

    if dataset is None and env is not None and teacher is not None:
        dataset = collect_expert_transitions(
            env, teacher, task=args.stage, num_episodes=args.num_episodes,
            seed=args.seed, stub=bool(args.stub),
        )
        print(f"[loglikelihood] collected {len(dataset)} expert samples on {args.stage}")
        if args.dataset:
            dataset.save(args.dataset)

    if dataset is None:
        print("[loglikelihood] no dataset available (need --dataset or --teacher + env)")
        return 2

    # -- evaluate --------------------------------------------------------
    tracker = LogLikelihoodTracker(
        dataset, batch_size=args.batch_size, device=args.device,
    )
    if teacher is not None:
        try:
            tracker.measure_reference(teacher)
        except Exception as exc:  # pragma: no cover
            print(f"[loglikelihood] reference measurement failed: {exc}")

    if args.checkpoint:
        try:
            from src.robotic_sequence.sac import build_sac_agent

            policy = _load_agent(build_sac_agent, args.checkpoint, env, cfg, args)
            tracker.record(0, policy)
        except Exception as exc:  # pragma: no cover
            print(f"[loglikelihood] could not evaluate checkpoint: {exc}")

    result = tracker.to_dict()
    print(json.dumps(result["summary"], indent=2, default=str))

    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(result, handle, default=str)
        print(f"[loglikelihood] wrote {args.output}")

    if args.plot:
        try:
            plot_loglikelihood(tracker, path=args.plot)
            print(f"[loglikelihood] wrote {args.plot}")
        except Exception as exc:  # pragma: no cover
            print(f"[loglikelihood] plotting failed: {exc}")

    if env is not None:
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
    return 0


def _load_agent(build_sac_agent: Any, path: str, env: Any, cfg: Any, args: Any) -> Any:
    """Reconstruct a SAC agent from a checkpoint (mirrors cka.py's CLI helper)."""
    obs_dim = getattr(env, "observation_dim", 9) if env is not None else 9
    action_dim = getattr(env, "action_dim", 4) if env is not None else 4
    n_stages = getattr(env, "n_stages", 4) if env is not None else 4
    try:
        from src.robotic_sequence.sac import SACAgent, SACPolicy, SACTwinQ

        if _HAS_TORCH:
            state = torch.load(path, map_location=args.device)
        else:  # pragma: no cover
            raise RuntimeError("PyTorch unavailable")
        if "policy" in state and isinstance(state["policy"], Mapping):
            policy = SACPolicy(obs_dim, action_dim, n_stages)
            policy.load_state_dict(state["policy"])
            q_net = SACTwinQ(obs_dim, action_dim, n_stages)
            if "q_network" in state and isinstance(state["q_network"], Mapping):
                q_net.load_state_dict(state["q_network"])
            agent = SACAgent(policy, q_net, device=args.device)
            return agent
    except Exception:
        pass
    agent = build_sac_agent(
        cfg, obs_dim, action_dim, n_stages=n_stages, device=args.device,
        seed=args.seed,
    )
    loader = getattr(agent, "load_pretrained", None)
    if callable(loader):
        loader(path)
    return agent


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
