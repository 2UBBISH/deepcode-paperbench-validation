"""Critical state identification for RICE (CORE COMPONENT #1, selection half).

The mask network :math:`\\tilde{\\pi}_{\\theta}` trained in
``rice.algorithms.mask_network`` (Algorithm 1) provides a *step-level explanation*
of the target agent's behaviour.  Quoting the paper (section 3.3, Technique
Detail, Step-level Explanation):

    "By applying this resolved mask to each state, we will be able to assess the
     state importance (i.e., the probability of mask network outputting "0") at
     any time step."

so, verbatim from the paper,

    importance(s_t) = P(a_t^m = 0 | s_t)

A *critical state* is the state of a trajectory that the mask network deems most
important.  Section 3.3 (Constructing Mixed Initial State Distribution):

    "Initially, we randomly sample a trajectory by executing the pre-trained
     policy pi.  Subsequently, the state mask is applied to pinpoint the most
     important state within the episode tau by assessing the significance of
     each state."

and, verbatim, Algorithm 2:

    "Run pi to obtain a trajectory tau of length K
     Identify the most critical state s_t in tau via state mask pi_tilde
     Set the initial state s_0 <- s_t"

Hence this module implements

    s_critical = argmax_{s_t in tau} P(a_t^m = 0 | s_t)

Nothing here inspects the internals of the target agent: the critical state is a
function of *visited states* and of the (separately trained) mask network only,
which keeps the black-box assumption required by the addendum.

Provided utilities
------------------
``importance_scores``               batched importance scorer P(mask = 0)
``select_critical_state``           argmax selection (state / index)
``select_top_k_critical_states``    the k most important states of a trajectory
``empirical_importance``            importance from realised mask actions 1 - mean(a^m)
``RollinTrajectory``                container for the length-K roll-in trajectory
``roll_in_trajectory``              run the pre-trained policy for K steps
``identify_critical_state``         roll-in + argmax (Algorithm 2 inner block)
``collect_critical_states``         pool of critical states over many roll-in episodes
``best_window_index`` / ``windowed_mean``   sliding-window segment used by the fidelity score
``CriticalStateSelector``           small convenience wrapper around the scorer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - torch is a hard requirement, kept defensive anyway
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .ppo import flatten_obs, make_target_policy_callable

__all__ = [
    "importance_scores",
    "select_critical_state",
    "select_top_k_critical_states",
    "empirical_importance",
    "RollinTrajectory",
    "roll_in_trajectory",
    "identify_critical_state",
    "collect_critical_states",
    "windowed_mean",
    "best_window_index",
    "CriticalStateSelector",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _resolve_device(device: Any) -> "torch.device":
    """Best-effort device resolution without importing private helpers."""
    if torch is None:  # pragma: no cover
        return "cpu"  # type: ignore[return-value]
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _state_to_array(state: Any) -> np.ndarray:
    """Flatten a single observation (Box or Dict) to a 1-D float32 array."""
    return np.asarray(flatten_obs(state), dtype=np.float32).reshape(-1)


def _batch_states(states: Sequence[Any], device: "torch.device") -> "torch.Tensor":
    arr = np.stack([_state_to_array(s) for s in states], axis=0).astype(np.float32)
    return torch.as_tensor(arr, device=device)


def _logits_of(mask_network: Any, obs: "torch.Tensor") -> "torch.Tensor":
    """Forward the mask network and return a ``(N, 2)`` tensor of logits."""
    logits = mask_network(obs)
    if isinstance(logits, tuple):  # some modules return (logits, value)
        logits = logits[0]
    logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.dim() == 1:
        logits = logits.reshape(-1, 2)
    return logits


def _single_importance(mask_network: Any, state: Any) -> float:
    """Importance of a single state (used only as a graceful fallback path)."""
    for attr in ("mask_prob_zero", "importance"):
        fn = getattr(mask_network, attr, None)
        if callable(fn):
            try:
                return float(np.asarray(fn(state)).reshape(-1)[0])
            except Exception:
                continue
    if torch is not None:
        device = _resolve_device(getattr(mask_network, "device", "auto"))
        with torch.no_grad():
            logits = _logits_of(mask_network, _batch_states([state], device))
            return float(torch.softmax(logits, dim=-1)[0, 0].item())
    raise TypeError(
        "mask_network must be a MaskNetwork-like module exposing "
        "'mask_prob_zero'/'importance', or a callable returning 2 logits."
    )


def set_eval_mode(mask_network: Any) -> Optional[bool]:
    """Put a ``torch.nn.Module`` in eval mode; return the previous mode if known."""
    if torch is not None and isinstance(mask_network, torch.nn.Module):
        was_training = mask_network.training
        mask_network.eval()
        return was_training
    return None


def restore_mode(mask_network: Any, was_training: Optional[bool]) -> None:
    if (
        torch is not None
        and was_training is not None
        and isinstance(mask_network, torch.nn.Module)
    ):
        mask_network.train(was_training)


# --------------------------------------------------------------------------------------
# importance scoring
# --------------------------------------------------------------------------------------
def importance_scores(
    mask_network: Any,
    states: Sequence[Any],
    batch_size: int = 512,
    device: Any = None,
) -> np.ndarray:
    """Importance of every state in ``states``: ``P(a_t^m = 0 | s_t)`` (paper, section 3.3).

    Parameters
    ----------
    mask_network:
        A trained ``MaskNetwork`` (or any object exposing ``mask_prob_zero`` /
        ``importance`` / a ``(N, 2)``-logit ``forward``).
    states:
        Sequence of *raw* observations (Box vectors or flattened Dicts).  The
        original observation objects are never modified.
    batch_size:
        Mini-batch size for the forward passes.
    device:
        Torch device override; defaults to the module's own device.

    Returns
    -------
    ``np.ndarray`` of shape ``(len(states),)`` and dtype ``float32`` holding
    ``P(mask = 0)`` for each state -- high values mark important/critical states.
    """
    states = list(states)
    if len(states) == 0:
        return np.zeros((0,), dtype=np.float32)

    # Already-computed importance scores are passed through unchanged.
    if not callable(mask_network) and isinstance(mask_network, (np.ndarray, list, tuple)):
        return np.asarray(mask_network, dtype=np.float32).reshape(-1)

    was_training = set_eval_mode(mask_network)
    try:
        module_device = getattr(mask_network, "device", None)
        dev = _resolve_device(device if device is not None else module_device)
        chunks: List[np.ndarray] = []
        for start in range(0, len(states), max(1, batch_size)):
            chunk = states[start : start + max(1, batch_size)]
            try:
                if torch is None:  # pragma: no cover
                    raise RuntimeError("torch unavailable")
                obs = _batch_states(chunk, dev)
                with torch.no_grad():
                    logits = _logits_of(mask_network, obs)
                    p_zero = torch.softmax(logits, dim=-1)[:, 0]
                chunks.append(p_zero.detach().cpu().numpy().astype(np.float32))
            except Exception:
                # graceful per-state fallback (e.g. exotic observation spaces)
                chunks.append(
                    np.asarray(
                        [_single_importance(mask_network, s) for s in chunk],
                        dtype=np.float32,
                    )
                )
        return np.concatenate(chunks, axis=0).astype(np.float32)
    finally:
        restore_mode(mask_network, was_training)


def empirical_importance(mask_actions: Sequence[int]) -> float:
    """Importance estimated from realised mask actions: ``1 - mean(a^m)``.

    Used as a sanity/consistency check: the *definition* of importance is
    ``P(mask = 0)``, so the empirical frequency of "keep" actions approximates it.
    """
    arr = np.asarray(mask_actions, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(1.0 - arr.mean())


def select_top_k_critical_states(
    mask_network: Any,
    states: Sequence[Any],
    k: int = 1,
    batch_size: int = 512,
    device: Any = None,
) -> Tuple[List[Any], List[int], np.ndarray]:
    """Return the ``k`` most important states (descending importance).

    Returns ``(states_topk, indices_topk, all_importance_scores)``.
    """
    states = list(states)
    scores = importance_scores(mask_network, states, batch_size=batch_size, device=device)
    if scores.size == 0:
        return [], [], scores
    k = int(max(1, min(k, scores.size)))
    order = np.argsort(-scores, kind="stable")[:k]
    idx = [int(i) for i in order]
    return [states[i] for i in idx], idx, scores


def select_critical_state(
    mask_network: Any,
    states: Sequence[Any],
    batch_size: int = 512,
    device: Any = None,
    return_info: bool = False,
):
    """Critical state of a trajectory: ``argmax_s P(mask = 0 | s)`` (Algorithm 2).

    Parameters
    ----------
    mask_network: trained mask network (or precomputed importance sequence).
    states: the visited states of trajectory ``tau`` (raw observations).
    return_info: when ``True`` also return
        ``(index, importance_value, all_importance_scores)``; ties are broken by
        the earliest time step (``np.argmax`` semantics), i.e. the first state
        attaining the maximal importance.

    Returns
    -------
    The critical state (raw observation), or the 4-tuple described above.
    """
    states = list(states)
    scores = importance_scores(mask_network, states, batch_size=batch_size, device=device)
    if scores.size == 0:
        raise ValueError("Cannot select a critical state from an empty trajectory.")
    index = int(np.argmax(scores))
    if return_info:
        return states[index], index, float(scores[index]), scores
    return states[index]


class CriticalStateSelector:
    """Convenience wrapper: score states and pick the most critical one.

    ``Beta``/reset probability ``p`` is handled by ``MixedInitSampler``; this class
    is deliberately limited to the *explanation* half (importance -> critical state).
    """

    def __init__(self, mask_network: Any, batch_size: int = 512, device: Any = None):
        self.mask_network = mask_network
        self.batch_size = int(batch_size)
        self.device = device

    # -- scoring -----------------------------------------------------------------
    def score(self, states: Sequence[Any]) -> np.ndarray:
        return importance_scores(
            self.mask_network, states, batch_size=self.batch_size, device=self.device
        )

    def best_index(self, scores: Sequence[float]) -> int:
        arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("Empty importance sequence.")
        return int(np.argmax(arr))

    # -- selection ---------------------------------------------------------------
    def select(self, states: Sequence[Any], return_info: bool = False):
        return select_critical_state(
            self.mask_network,
            states,
            batch_size=self.batch_size,
            device=self.device,
            return_info=return_info,
        )

    def select_top_k(self, states: Sequence[Any], k: int = 1):
        return select_top_k_critical_states(
            self.mask_network, states, k=k, batch_size=self.batch_size, device=self.device
        )

    def importance(self, state: Any) -> float:
        """Importance of a single state (used by tests / qualitative analysis)."""
        return float(self.score([state])[0])


# --------------------------------------------------------------------------------------
# sliding window (shared with the fidelity score, Experiment I)
# --------------------------------------------------------------------------------------
def windowed_mean(scores: Sequence[float], window: int) -> np.ndarray:
    """Sliding-window average of ``scores`` with width ``window`` (valid windows).

    The paper (section 4.1) describes stepping a sliding window through the
    trajectory "and then choose the window with the highest average importance
    score".  Returns an array of length ``len(scores) - window + 1`` (or a single
    value when ``window >= len(scores)``, which corresponds to the whole
    trajectory).
    """
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    window = int(window)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if window <= 1:
        return arr.copy()
    if window >= arr.size:
        return np.asarray([arr.mean()], dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    return (csum[window:] - csum[:-window]) / float(window)


def best_window_index(scores: Sequence[float], window: int) -> int:
    """Start index of the window with the highest average importance score."""
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    window = int(window)
    if arr.size == 0:
        raise ValueError("Empty importance sequence.")
    if window >= arr.size:
        return 0
    return int(np.argmax(windowed_mean(arr, window)))


# --------------------------------------------------------------------------------------
# roll-in trajectory (length K) used by Algorithm 2
# --------------------------------------------------------------------------------------
def max_episode_steps(env: Any, default: int = 1000) -> int:
    """Best-effort lookup of an environment's episode length limit."""
    candidates = []
    spec = getattr(env, "spec", None)
    if spec is not None:
        candidates.append(getattr(spec, "max_episode_steps", None))
    for holder in (env, getattr(env, "unwrapped", None)):
        if holder is None:
            continue
        candidates.append(getattr(holder, "_max_episode_steps", None))
        candidates.append(getattr(holder, "max_episode_steps", None))
    for value in candidates:
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _env_reset(env: Any, seed: Optional[int] = None) -> Any:
    """Reset supporting both gym (obs) and gymnasium ((obs, info)) signatures."""
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        if seed is not None:
            try:
                env.seed(seed)
            except Exception:
                pass
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _env_step(env: Any, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """Step supporting both 4-tuple (gym) and 5-tuple (gymnasium) returns."""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
    else:
        obs, reward, done, info = out
        done = bool(done)
    return obs, float(reward), done, dict(info) if info is not None else {}


@dataclass
class RollinTrajectory:
    """A trajectory ``tau`` of length ``K`` produced by the pre-trained policy.

    Fields mirror Algorithm 2's roll-in block; ``states`` are the *raw*
    observations so that the mask network can be queried on them (or they can be
    handed to a simulator state-restore manager).
    """

    states: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[Any] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)

    # -- explanation ------------------------------------------------------------
    def importances(self, mask_network: Any, **kwargs: Any) -> np.ndarray:
        return importance_scores(mask_network, self.states, **kwargs)

    def critical_index(self, mask_network: Any, **kwargs: Any) -> int:
        """Index of the most critical step: ``argmax_t P(mask = 0 | s_t)``."""
        scores = self.importances(mask_network, **kwargs)
        if scores.size == 0:
            raise ValueError("Empty roll-in trajectory.")
        return int(np.argmax(scores))

    def critical_state(self, mask_network: Any, return_index: bool = False, **kwargs: Any):
        idx = self.critical_index(mask_network, **kwargs)
        if return_index:
            return self.states[idx], idx
        return self.states[idx]


def roll_in_trajectory(
    env: Any,
    policy: Any,
    length: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    reset: bool = True,
    seed: Optional[int] = None,
    initial_state: Any = None,
    state_manager: Any = None,
    obs: Any = None,
) -> RollinTrajectory:
    """Run the (pre-trained) policy ``pi`` to obtain a trajectory ``tau`` of length ``K``.

    Implements the first two lines of Algorithm 2's ``RAND_NUM < p`` branch::

        Run pi to obtain a trajectory tau of length K
        Identify the most critical state s_t in tau via state mask pi_tilde

    The trajectory is *not* used for learning here: it only serves to locate the
    critical state that becomes the next initial state.

    Parameters
    ----------
    env: environment to interact with.
    policy: SB3 model, ``ActorCritic``/``MaskNetwork``-like module, or callable
        ``obs -> action`` (see ``make_target_policy_callable``).
    length: ``K``, the roll-in horizon.  Defaults to one full pre-trained-policy
        episode (``max_episode_steps``), per the plan's "K -> one full
        pre-trained-policy episode".
    rng: ``numpy`` generator used only for tie-free random action sampling.
    reset: if ``True`` (default) the environment is reset to ``s_0 ~ rho`` first;
        otherwise the current simulator state is used.
    seed: optional reset seed.
    initial_state: an already-identified critical state.  When a
        ``state_manager`` is supplied the simulator is restored to it
        (Go-Explore style, ``rice.algorithms.env_reset``); otherwise, for
        environments whose observation *is* the full state (e.g. classic
        control), the observation itself is returned as the starting point and
        no reset is performed.
    state_manager: optional ``EnvStateManager`` used to restore ``initial_state``.
    obs: optional current observation to start from (used when ``reset=False``
        and the caller already holds the raw observation).
    """
    policy_fn = make_target_policy_callable(policy)
    horizon = int(length) if length is not None else max_episode_steps(env)

    if state_manager is not None and initial_state is not None:
        state_manager.restore(initial_state)
        if obs is None:
            obs = getattr(state_manager, "current_observation", None)
    if obs is None and reset:
        obs = _env_reset(env, seed=seed)
    if obs is None and initial_state is not None and not reset:
        # Observation-as-state environments (MuJoCo classic control): the raw
        # observation is a valid stand-in for the simulator state.
        obs = initial_state
    if obs is None:
        raise ValueError(
            "roll_in_trajectory needs an initial observation: pass reset=True, "
            "obs=..., or (initial_state + state_manager)."
        )

    traj = RollinTrajectory()
    for _ in range(max(1, horizon)):
        action = np.asarray(policy_fn(obs))
        next_obs, reward, done, info = _env_step(env, action)
        traj.states.append(obs)
        traj.actions.append(action)
        traj.rewards.append(reward)
        traj.next_states.append(next_obs)
        traj.dones.append(done)
        traj.infos.append(info)
        if done:
            break
        obs = next_obs
    return traj


def identify_critical_state(
    env: Any,
    policy: Any,
    mask_network: Any,
    length: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    reset: bool = True,
    seed: Optional[int] = None,
    return_trajectory: bool = False,
    batch_size: int = 512,
    device: Any = None,
):
    """Full Algorithm-2 roll-in block: roll out ``pi`` for K steps, return ``argmax`` state.

    Returns the critical state, or ``(critical_state, index, trajectory)`` when
    ``return_trajectory=True``.
    """
    traj = roll_in_trajectory(
        env=env, policy=policy, length=length, rng=rng, reset=reset, seed=seed
    )
    if len(traj) == 0:
        raise RuntimeError("Roll-in trajectory is empty; cannot identify a critical state.")
    scores = traj.importances(mask_network, batch_size=batch_size, device=device)
    index = int(np.argmax(scores))
    if return_trajectory:
        return traj.states[index], index, traj
    return traj.states[index]


def collect_critical_states(
    env: Any,
    policy: Any,
    mask_network: Any,
    n_trajectories: int = 1,
    length: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    batch_size: int = 512,
    device: Any = None,
    progress: bool = False,
) -> List[Any]:
    """Sample ``n_trajectories`` roll-in episodes and keep each episode's critical state.

    The resulting list is an empirical sample of the distribution
    ``d_rho^{pi_hat}(s)`` of identified critical states that the mixed initial state
    distribution ``mu(s) = beta * d_rho^{pi_hat}(s) + (1 - beta) * rho(s)`` draws from
    (section 3.3, Constructing Mixed Initial State Distribution).
    """
    critical: List[Any] = []
    for i in range(int(n_trajectories)):
        ep_seed = None if seed is None else int(seed) + i
        state = identify_critical_state(
            env=env,
            policy=policy,
            mask_network=mask_network,
            length=length,
            rng=rng,
            reset=True,
            seed=ep_seed,
            batch_size=batch_size,
            device=device,
        )
        critical.append(state)
        if progress and (i + 1) % 10 == 0:  # pragma: no cover - cosmetic
            print(f"[critical_state] collected {i + 1}/{n_trajectories}")
    return critical
