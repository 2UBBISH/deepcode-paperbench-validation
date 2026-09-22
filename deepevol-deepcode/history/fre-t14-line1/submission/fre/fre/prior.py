"""Prior reward distribution ``p(eta)`` for Functional Reward Encoding (FRE).

This module implements Component 1 of the FRE reproduction plan: the mixture of
random unsupervised reward functions used to train the FRE encoder/decoder
(paper Section 4.2 "Random Functions as a Prior Reward Distribution" and
Appendix B "Training Details"; addendum "Clarifications on FRE Prior Reward
Distributions").

The prior is a uniform mixture of three function classes:

1. **Singleton / goal-reaching** functions -- "reward = -1 for every timestep
   where the goal has not been reached, and 0 otherwise" (Appendix B).  Goals
   are sampled with the hindsight-experience-relabelling (HER) distribution of
   (Park et al., 2023a): for a randomly selected state we use it as the goal
   with probability 0.2, a *future* state within the same trajectory with
   probability 0.5, and a completely random dataset state with probability 0.3.
   A done mask is set to True when the goal is achieved and we ensure that at
   least one of the encoder samples *is* the goal state (Appendix B).

2. **Random linear** functions -- "the inner product between a uniformly random
   vector and the current state" (Section 4.2; Appendix B).  The vector is
   uniform in ``[-1, 1]``.  A random binary mask zeroes each dimension with
   probability 0.9 ("to encourage sparsity and bias towards simpler
   functions").  On AntMaze the XY positions are *removed* from this generation
   because "the scale of the dimensions led to instability" (Appendix B).

3. **Random MLP** functions -- a neural network of size ``(state_dim, 32, 1)``
   with parameters drawn from a normal distribution "scaled by the average
   dimension of the layer", a ``tanh`` activation between the two layers and a
   final output clipped to ``[-1, 1]`` (Appendix B).

The single entry point used by the trainer is :class:`PriorSampler`, which
samples a batch of reward functions (one per batch element), the corresponding
encoder/decoder states and the evaluated rewards/done-masks:

    >>> sampler = PriorSampler.from_config(cfg, state_dim=29)
    >>> batch = sampler.sample_batch(dataset, batch_size=512)
    >>> batch.encoder_rewards.shape     # (512, 32)
    >>> batch.decoder_rewards.shape     # (512, 8)

Class indices of the sampled functions are returned in ``batch.function_types``
so the trainer can log the realised mixture (and so that the prior ablations of
Table 4 -- FRE-goals / FRE-lin / FRE-mlp / FRE-lin-mlp / FRE-goal-mlp /
FRE-goal-lin -- can be realised simply by changing the mixture ratios).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "PriorBatch",
    "GoalReachingPrior",
    "LinearPrior",
    "MLPPrior",
    "PriorSampler",
    "make_prior_sampler",
    "ArrayPriorDataset",
    "PRIOR_CLASS_NAMES",
    "ABLATION_MIXTURES",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: canonical ordering of the three prior function classes
PRIOR_CLASS_NAMES: Tuple[str, ...] = ("goal", "linear", "mlp")

#: Table 4 / addendum prior ablations expressed as mixtures.  ``FRE-all`` is the
#: vanilla uniform mixture used in Sections 5.1-5.4.
ABLATION_MIXTURES: Dict[str, Dict[str, float]] = {
    "all": {"goal": 1.0 / 3.0, "linear": 1.0 / 3.0, "mlp": 1.0 / 3.0},
    "goals": {"goal": 1.0},
    "lin": {"linear": 1.0},
    "mlp": {"mlp": 1.0},
    "lin-mlp": {"linear": 0.5, "mlp": 0.5},
    "goal-mlp": {"goal": 0.5, "mlp": 0.5},
    "goal-lin": {"goal": 0.5, "linear": 0.5},
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_tensor(value: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert ``value`` (tensor / ndarray / sequence) to a float tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def _normalize_ratios(ratios: Any) -> Tuple[List[str], List[float]]:
    """Normalise a prior-ratio specification into ordered (names, probs).

    Accepts a dict (``{"goal": 0.33, ...}``), a sequence of names with equal
    weights, or an existing ``(names, probs)`` pair.  Unknown names raise
    ``KeyError`` so typos surface immediately.
    """
    if ratios is None:
        ratios = dict(ABLATION_MIXTURES["all"])
    if isinstance(ratios, tuple) and len(ratios) == 2 and isinstance(ratios[0], (list, tuple)):
        names, probs = list(ratios[0]), [float(p) for p in ratios[1]]
    elif isinstance(ratios, dict):
        names = [str(k) for k in ratios.keys()]
        probs = [float(v) for v in ratios.values()]
    else:  # iterable of names -> equal weights
        names = [str(k) for k in ratios]
        probs = [1.0 / len(names)] * len(names)
    for name in names:
        if name not in PRIOR_CLASS_NAMES:
            raise KeyError(f"unknown prior class {name!r}; expected one of {PRIOR_CLASS_NAMES}")
    names = [n for n in PRIOR_CLASS_NAMES if n in names]  # canonical order
    probs = [probs[0] for _ in names] if len(set(names)) != len(probs) else probs
    # re-order probabilities to match the canonical name order
    if isinstance(ratios, dict):
        probs = [float(ratios[n]) for n in names]
    elif not (isinstance(ratios, tuple) and len(ratios) == 2 and isinstance(ratios[0], (list, tuple))):
        probs = [1.0 / len(names)] * len(names)
    total = float(sum(probs))
    if total <= 0.0:
        raise ValueError("prior ratios must sum to a positive value")
    probs = [p / total for p in probs]
    return names, probs


# ---------------------------------------------------------------------------
# Dataset access helpers (duck-typed against rl/replay_buffer.py)
# ---------------------------------------------------------------------------


def _dataset_state_dim(dataset: Any) -> Optional[int]:
    for attr in ("state_dim", "obs_dim", "observation_dim", "num_state_dims"):
        value = getattr(dataset, attr, None)
        if isinstance(value, int):
            return value
    for attr in ("states", "observations", "obs"):
        value = getattr(dataset, attr, None)
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            return int(value.shape[-1])
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return int(value.shape[-1])
    return None


def _dataset_state_std(dataset: Any) -> Optional[torch.Tensor]:
    """Per-dimension standard deviation of the dataset (used for ExORL goals).

    Appendix C.2: "Each state dimension is normalized according to the standard
    deviation along that dimension within the offline dataset."
    """
    if dataset is None:
        return None
    for attr in ("state_std", "obs_std", "std", "dim_scale", "dimension_scale"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return _to_tensor(value, torch.device("cpu")).reshape(-1)
    for attr in ("states", "observations", "obs"):
        value = getattr(dataset, attr, None)
        if value is not None:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim >= 2:
                std = arr.reshape(-1, arr.shape[-1]).std(axis=0)
                return torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32)
    return None


def _sample_states(dataset: Any, num: int, device: torch.device) -> torch.Tensor:
    """Sample ``num`` random states (shape ``(num, state_dim)``) from a dataset."""
    for name in ("sample_states", "sample_observations", "sample_state", "sample_obs"):
        fn = getattr(dataset, name, None)
        if callable(fn):
            return _to_tensor(fn(num), device)
    for name in ("sample_batch", "sample", "sample_batches"):
        fn = getattr(dataset, name, None)
        if callable(fn):
            out = fn(num)
            return _unpack_batch_states(out, device)
    for attr in ("states", "observations", "obs"):
        value = getattr(dataset, attr, None)
        if value is not None:
            arr = _to_tensor(value, device)
            idx = torch.randint(0, arr.shape[0], (num,), device=device)
            return arr[idx]
    raise TypeError(
        "cannot sample states from dataset of type "
        f"{type(dataset).__name__}; expected sample_states(n) or a .states array"
    )


def _unpack_batch_states(out: Any, device: torch.device) -> torch.Tensor:
    """Flatten a dict/tuple batch into ``(n, state_dim)`` states."""
    if isinstance(out, dict):
        for key in ("observations", "states", "obs", "state", "observation"):
            if key in out:
                arr = _to_tensor(out[key], device)
                return arr.reshape(-1, arr.shape[-1])
        raise KeyError(f"cannot find states in batch dict with keys {sorted(out.keys())}")
    if isinstance(out, (tuple, list)) and out:
        arr = _to_tensor(out[0], device)
        return arr.reshape(-1, arr.shape[-1])
    arr = _to_tensor(out, device)
    return arr.reshape(-1, arr.shape[-1])


def _sample_trajectories(dataset: Any, num: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ``num`` trajectories -> ``(states (n, T, D), valid_mask (n, T))``.

    Falls back to sampling flat states (each treated as a length-1 trajectory)
    when the dataset does not expose trajectory-level sampling.
    """
    for name in ("sample_trajectories", "sample_trajectory_batch", "sample_trajs", "sample_trajectory"):
        fn = getattr(dataset, name, None)
        if callable(fn):
            return _unpack_trajectories(fn(num), device)
    # try a generic sample() that may return sequences
    for name in ("sample_batch", "sample"):
        fn = getattr(dataset, name, None)
        if callable(fn):
            out = fn(num)
            if isinstance(out, dict):
                for key in ("observations", "states", "obs", "state"):
                    if key in out:
                        arr = _to_tensor(out[key], device)
                        if arr.ndim == 3:
                            mask = _mask_from_batch(out, arr, device)
                            return arr, mask
            elif isinstance(out, (tuple, list)) and out:
                arr = _to_tensor(out[0], device)
                if arr.ndim == 3:
                    mask = _mask_from_batch(out, arr, device)
                    return arr, mask
    states = _sample_states(dataset, num, device)
    mask = torch.ones(states.shape[0], 1, device=device, dtype=torch.bool)
    return states.unsqueeze(1), mask


def _mask_from_batch(out: Any, arr: torch.Tensor, device: torch.device) -> torch.Tensor:
    n, T, _ = arr.shape
    mask = torch.ones(n, T, device=device, dtype=torch.bool)
    if isinstance(out, dict):
        for key in ("terminals", "dones", "done", "terminal", "masks", "mask"):
            if key in out:
                flags = _to_tensor(out[key], device).reshape(n, T) > 0.5
                if key in ("masks", "mask"):
                    return flags
                mask = ~flags
                if "timeouts" in out:  # time-limit truncation is not a real done
                    timeouts = _to_tensor(out["timeouts"], device).reshape(n, T) > 0.5
                    mask = mask | timeouts
                return mask
    return mask


def _unpack_trajectories(out: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(out, dict):
        for key in ("observations", "states", "obs", "state"):
            if key in out:
                arr = _to_tensor(out[key], device)
                if arr.ndim == 2:  # (T, D) single trajectory
                    arr = arr.unsqueeze(0)
                return arr, _mask_from_batch(out, arr, device)
        raise KeyError(f"cannot find states in trajectory dict with keys {sorted(out.keys())}")
    if isinstance(out, (tuple, list)):
        arr = _to_tensor(out[0], device)
        if arr.ndim == 2:
            arr = arr.unsqueeze(0)
        mask = torch.ones(arr.shape[0], arr.shape[1], device=device, dtype=torch.bool)
        if len(out) > 1 and out[1] is not None:
            flags = _to_tensor(out[1], device)
            if flags.numel() and flags.dtype in (torch.bool, torch.float32, torch.float64):
                flags = flags.reshape(arr.shape[0], arr.shape[1])
                mask = ~(flags > 0.5)
        return arr, mask
    arr = _to_tensor(out, device)
    if arr.ndim == 2:
        arr = arr.unsqueeze(0)
    return arr, torch.ones(arr.shape[0], arr.shape[1], device=device, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------


class PriorBatch:
    """Container for one batch of sampled reward functions and their values.

    Attributes
    ----------
    encoder_states / encoder_rewards / encoder_dones: ``(B, K, D)`` / ``(B, K)`` / ``(B, K)``
        The ``K = 32`` state-reward pairs that form the encoder input lookup
        table ``L_eta^e`` (Section 4.1, Eq. 6).
    decoder_states / decoder_rewards / decoder_dones: ``(B, K', D)`` / ``(B, K')`` / ``(B, K')``
        The ``K' = 8`` held-out pairs used for the reconstruction term of Eq. 6.
    function_types: ``(B,)`` long tensor with indices into :data:`PRIOR_CLASS_NAMES`
    function_names: list of ``B`` strings (human readable class of each row)
    params: dict of per-class sampled parameters (e.g. the HER goals)
    """

    __slots__ = (
        "encoder_states",
        "encoder_rewards",
        "encoder_dones",
        "decoder_states",
        "decoder_rewards",
        "decoder_dones",
        "function_types",
        "function_names",
        "params",
    )

    def __init__(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        encoder_dones: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
        decoder_dones: torch.Tensor,
        function_types: torch.Tensor,
        function_names: Sequence[str],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.encoder_states = encoder_states
        self.encoder_rewards = encoder_rewards
        self.encoder_dones = encoder_dones
        self.decoder_states = decoder_states
        self.decoder_rewards = decoder_rewards
        self.decoder_dones = decoder_dones
        self.function_types = function_types
        self.function_names = list(function_names)
        self.params = dict(params or {})

    # -- convenience ------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return int(self.encoder_rewards.shape[0])

    @property
    def num_encoder_samples(self) -> int:
        return int(self.encoder_rewards.shape[1])

    @property
    def num_decoder_samples(self) -> int:
        return int(self.decoder_rewards.shape[1])

    def mixture_counts(self) -> Dict[str, int]:
        """Number of rows per prior class actually sampled in this batch."""
        counts: Dict[str, int] = {name: 0 for name in PRIOR_CLASS_NAMES}
        for idx in self.function_types.tolist():
            counts[PRIOR_CLASS_NAMES[int(idx)]] += 1
        return counts

    def to(self, device: torch.device) -> "PriorBatch":
        return PriorBatch(
            encoder_states=self.encoder_states.to(device),
            encoder_rewards=self.encoder_rewards.to(device),
            encoder_dones=self.encoder_dones.to(device),
            decoder_states=self.decoder_states.to(device),
            decoder_rewards=self.decoder_rewards.to(device),
            decoder_dones=self.decoder_dones.to(device),
            function_types=self.function_types.to(device),
            function_names=self.function_names,
            params={k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.params.items()},
        )

    def as_dict(self) -> Dict[str, Any]:
        """Scalar diagnostics for logging (counts + reward statistics)."""
        with torch.no_grad():
            enc = self.encoder_rewards
            dec = self.decoder_rewards
            return {
                "prior/encoder_reward_mean": float(enc.mean()),
                "prior/encoder_reward_min": float(enc.min()),
                "prior/encoder_reward_max": float(enc.max()),
                "prior/decoder_reward_mean": float(dec.mean()),
                "prior/encoder_done_rate": float(self.encoder_dones.mean()),
                "prior/decoder_done_rate": float(self.decoder_dones.mean()),
                "prior/count_goal": int((self.function_types == 0).sum()),
                "prior/count_linear": int((self.function_types == 1).sum()),
                "prior/count_mlp": int((self.function_types == 2).sum()),
            }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        counts = self.mixture_counts()
        return (
            f"PriorBatch(batch_size={self.batch_size}, K={self.num_encoder_samples}, "
            f"K'={self.num_decoder_samples}, counts={counts})"
        )


# ---------------------------------------------------------------------------
# 1) Singleton / goal-reaching reward functions
# ---------------------------------------------------------------------------


class GoalReachingPrior:
    """Singleton "goal-reaching" reward functions (Appendix B, first bullet).

    Reward definition (paper verbatim): "Reward is set to ``-1`` for every
    timestep that the goal is not achieved. A done mask is set to ``True`` when
    the goal is achieved."

    Goals are drawn from the HER distribution of (Park et al., 2023a): given a
    randomly selected state, it is used as the goal with probability 0.2, a
    future state within the same trajectory with probability 0.5, and a
    completely random dataset state with probability 0.3.

    Parameters
    ----------
    state_dim: dimensionality of the (possibly physics-augmented) state.
    threshold: goal distance below which the goal counts as reached.  The paper
        specifies ``2`` (XY distance) for AntMaze eval tasks and ``0.1``
        (std-normalised Euclidean distance) for ExORL; the prior uses the same
        metric as the downstream tasks of the domain.  Defaults to 0.5.
    scale: optional per-dimension normaliser (dataset std) used for ExORL.
    distance_dims: optional index tuple/slice restricting the distance to the
        base observation dimensions ("Augmented information is not utilized
        when calculating goal distance", Appendix C.2).
    distance_type: ``"euclidean"`` (AntMaze / ExORL) or ``"discrete"``
        (Kitchen-style discrete sub-task state matching).
    """

    name = "goal"
    class_index = 0

    def __init__(
        self,
        state_dim: int,
        threshold: float = 0.5,
        scale: Optional[torch.Tensor] = None,
        distance_dims: Optional[Sequence[int]] = None,
        distance_type: str = "euclidean",
        her_probs: Tuple[float, float, float] = (0.2, 0.5, 0.3),
        discrete_tol: float = 1e-2,
        force_goal_in_encoder: bool = True,
        reward_on_success: float = 0.0,
        reward_off_success: float = -1.0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.threshold = float(threshold)
        self.distance_type = str(distance_type)
        self.discrete_tol = float(discrete_tol)
        self.force_goal_in_encoder = bool(force_goal_in_encoder)
        self.reward_on_success = float(reward_on_success)
        self.reward_off_success = float(reward_off_success)
        probs = np.asarray(her_probs, dtype=np.float64)
        if probs.shape != (3,) or probs.sum() <= 0:
            raise ValueError("her_probs must be three non-negative numbers summing to > 0")
        self.her_probs = probs / probs.sum()
        if distance_dims is None:
            self.distance_dims = None
        else:
            self.distance_dims = tuple(int(d) for d in distance_dims)
        scale_t = None if scale is None else _to_tensor(scale, torch.device("cpu")).reshape(-1)
        if scale_t is not None and scale_t.numel() != self.state_dim and self.distance_dims is None:
            # A shorter scale implies only the leading dims are normalised.
            if scale_t.numel() < self.state_dim:
                pad = torch.ones(self.state_dim - scale_t.numel())
                scale_t = torch.cat([scale_t, pad], dim=0)
        self.scale = scale_t

    # -- parameters -------------------------------------------------------
    def sample_params(
        self,
        batch_size: int,
        dataset: Any = None,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample ``batch_size`` HER goals (shape ``(batch_size, state_dim)``)."""
        device = device or torch.device("cpu")
        rng = rng or np.random.default_rng()
        num = int(batch_size)
        goals = torch.zeros(num, self.state_dim, device=device, dtype=torch.float32)
        current = torch.zeros(num, self.state_dim, device=device, dtype=torch.float32)

        if dataset is None:
            raise ValueError(
                "GoalReachingPrior.sample_params requires a dataset to sample HER goals; "
                "pass encoder/decoder states to PriorSampler.sample_rewards instead."
            )

        traj_states, traj_mask = _sample_trajectories(dataset, num, device)
        n, T, _ = traj_states.shape
        T = max(T, 1)

        # choose a random (valid) "current" timestep per trajectory
        u_cur = torch.rand(n, generator=generator, device="cpu", dtype=torch.float32)
        if traj_mask.any():
            idx_grid = torch.arange(T, device=device).expand(n, T)
            valid = traj_mask
            # randint-style choice among valid indices
            scores = torch.where(valid, torch.rand(n, T, generator=generator, device="cpu", device=device), torch.zeros(n, T, device=device))
            t_cur = scores.argmax(dim=1)
        else:
            t_cur = (u_cur * T).long().to(device).clamp(max=T - 1)

        current = traj_states.gather(1, t_cur.view(n, 1, 1).expand(n, 1, self.state_dim)).squeeze(1)
        goals = current.clone()

        # future indices: uniform in [t_cur, T-1]
        u_future = torch.rand(n, generator=generator, device="cpu")
        span = (T - 1 - t_cur.cpu()).clamp(min=0)
        t_future = t_cur.cpu() + (u_future * (span + 1).float()).floor().long()
        t_future = t_future.clamp(max=T - 1).to(device)
        future = traj_states.gather(1, t_future.view(n, 1, 1).expand(n, 1, self.state_dim)).squeeze(1)

        # random dataset states
        rand_states = _sample_states(dataset, num, device)
        if rand_states.shape[-1] != self.state_dim:
            rand_states = self._fit_dim(rand_states, device)

        # mixture: 0.2 current, 0.5 future, 0.3 random
        u = rng.random(num)
        pick_current = u < self.her_probs[0]
        pick_future = (u >= self.her_probs[0]) & (u < self.her_probs[0] + self.her_probs[1])
        pick_random = ~(pick_current | pick_future)

        pc = torch.as_tensor(pick_current, device=device)
        pf = torch.as_tensor(pick_future, device=device)
        pr = torch.as_tensor(pick_random, device=device)
        goals = torch.where(pc.view(-1, 1), current, goals)
        goals = torch.where(pf.view(-1, 1), future, goals)
        goals = torch.where(pr.view(-1, 1), rand_states, goals)

        return {"goal": goals.detach(), "her_current_states": current.detach()}

    def _fit_dim(self, states: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Pad/truncate states to ``state_dim`` (datasets may append physics info)."""
        d = states.shape[-1]
        if d == self.state_dim:
            return states
        if d > self.state_dim:
            return states[..., : self.state_dim]
        pad = torch.zeros(*states.shape[:-1], self.state_dim - d, device=device)
        return torch.cat([states, pad], dim=-1)

    # -- evaluation -------------------------------------------------------
    def _distance(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        goals = self._fit_dim(goals, states.device)
        diff = states - goals.unsqueeze(-2)
        if self.distance_dims is not None:
            idx = torch.as_tensor(self.distance_dims, device=diff.device, dtype=torch.long)
            diff = diff.index_select(-1, idx)
        if self.scale is not None and self.distance_type == "euclidean":
            scale = self.scale.to(diff.device)
            if self.distance_dims is not None:
                idx = torch.as_tensor(self.distance_dims, device=diff.device, dtype=torch.long)
                scale = scale.index_select(0, idx)
            elif scale.numel() == diff.shape[-1]:
                pass
            elif scale.numel() > diff.shape[-1]:
                scale = scale[: diff.shape[-1]]
            else:
                pad = torch.ones(diff.shape[-1] - scale.numel(), device=diff.device)
                scale = torch.cat([scale, pad], dim=0)
            diff = diff / scale.clamp(min=1e-8)
        if self.distance_type == "discrete":
            return diff.abs().max(dim=-1).values
        return torch.linalg.vector_norm(diff, dim=-1)

    def reached(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Boolean ``(B, N)`` mask of states that have reached their goal."""
        return self._distance(states, goals) < self.threshold

    def rewards(self, states: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Reward ``-1`` unless the goal is reached, else ``0`` (paper verbatim)."""
        reached = self.reached(states, params["goal"])
        return torch.where(
            reached,
            torch.full_like(reached, self.reward_on_success, dtype=states.dtype),
            torch.full_like(reached, self.reward_off_success, dtype=states.dtype),
        )

    def dones(self, states: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Done mask: True where the goal has been achieved (Appendix B)."""
        return self.reached(states, params["goal"])

    def evaluate(self, states: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        reached = self.reached(states, params["goal"])
        rewards = torch.where(
            reached,
            torch.full_like(reached, self.reward_on_success, dtype=states.dtype),
            torch.full_like(reached, self.reward_off_success, dtype=states.dtype),
        )
        return rewards, reached


# ---------------------------------------------------------------------------
# 2) Random linear reward functions
# ---------------------------------------------------------------------------


class LinearPrior:
    """Random linear reward functions (Section 4.2, Appendix B).

    "Random Linear functions are generated according to a uniform vector within
    ``-1`` and ``1``. On AntMaze, we remove the XY positions from this
    generation as the scale of the dimensions led to instability. A random
    binary mask is applied with a ``0.9`` chance to zero the vector at that
    dimension, to encourage sparsity and bias towards simpler functions."
    """

    name = "linear"
    class_index = 1

    def __init__(
        self,
        state_dim: int,
        exclude_dims: Sequence[int] = (),
        mask_zero_prob: float = 0.9,
        low: float = -1.0,
        high: float = 1.0,
        clip_rewards: Optional[float] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.exclude_dims = tuple(int(d) for d in exclude_dims)
        self.mask_zero_prob = float(mask_zero_prob)
        self.low = float(low)
        self.high = float(high)
        self.clip_rewards = clip_rewards

    def sample_params(
        self,
        batch_size: int,
        dataset: Any = None,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        device = device or torch.device("cpu")
        rng = rng or np.random.default_rng()
        b = int(batch_size)
        u = rng.uniform(self.low, self.high, size=(b, self.state_dim)).astype(np.float32)
        weights = torch.as_tensor(u, device=device)
        # binary mask: each dim zeroed with probability `mask_zero_prob`
        keep = (rng.random((b, self.state_dim)) >= self.mask_zero_prob).astype(np.float32)
        weights = weights * torch.as_tensor(keep, device=device)
        if self.exclude_dims:
            idx = torch.as_tensor(self.exclude_dims, device=device, dtype=torch.long)
            weights.index_fill_(1, idx, 0.0)
        return {"weight": weights.detach()}

    def rewards(self, states: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Inner product between the (masked) random vector and the state."""
        weight = params["weight"].to(states.device, states.dtype)
        out = torch.einsum("bnd,bd->bn", states, weight)
        if self.clip_rewards is not None:
            out = out.clamp(-self.clip_rewards, self.clip_rewards)
        return out

    def dones(self, states: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        return torch.zeros(states.shape[:-1], dtype=torch.bool, device=states.device)

    def evaluate(self, states: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rewards(states, params), self.dones(states, params)


# ---------------------------------------------------------------------------
# 3) Random MLP reward functions
# ---------------------------------------------------------------------------


class MLPPrior:
    """Random MLP reward functions (Section 4.2, Appendix B).

    "Random MLP functions are generated using a neural network of size
    ``(state_dim, 32, 1)``. Parameters are sampled using a normal distribution
    scaled by the average dimension of the layer. A ``tanh`` activation is used
    between the two layers. The final output of the neural network is clipped
    between ``-1`` and ``1``."

    ``scale_mode``:
      * ``"inverse_sqrt"`` (default) -- ``std = 1 / sqrt(avg_dim)``; keeps
        pre-activations O(1) so ``tanh`` is genuinely nonlinear, matching the
        paper's description of MLPs as "intermediate function complexity".
      * ``"inverse"`` -- ``std = 1 / avg_dim``, the literal reading of
        "scaled by the average dimension".
    """

    name = "mlp"
    class_index = 2

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 32,
        output_clip: float = 1.0,
        activation: str = "tanh",
        scale_mode: str = "inverse_sqrt",
        random_bias: bool = False,
    ) -> None:
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_clip = float(output_clip)
        self.activation = str(activation)
        self.scale_mode = str(scale_mode)
        self.random_bias = bool(random_bias)

    def _std(self, fan_in: int, fan_out: int) -> float:
        avg = 0.5 * (fan_in + fan_out)
        if self.scale_mode == "inverse":
            return 1.0 / max(avg, 1e-6)
        return 1.0 / math.sqrt(max(avg, 1e-6))

    def sample_params(
        self,
        batch_size: int,
        dataset: Any = None,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        device = device or torch.device("cpu")
        gen = generator
        b = int(batch_size)
        d, h = self.state_dim, self.hidden_dim
        std1 = self._std(d, h)
        std2 = self._std(h, 1)
        w1 = torch.randn(b, h, d, generator=gen) * std1
        w2 = torch.randn(b, 1, h, generator=gen) * std2
        if self.random_bias:
            b1 = torch.randn(b, h, generator=gen) * std1
            b2 = torch.randn(b, 1, generator=gen) * std2
        else:
            b1 = torch.zeros(b, h)
            b2 = torch.zeros(b, 1)
        return {
            "w1": w1.to(device),
            "b1": b1.to(device),
            "w2": w2.to(device),
            "b2": b2.to(device),
        }

    def rewards(self, states: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        w1 = params["w1"].to(states.device, states.dtype)
        b1 = params["b1"].to(states.device, states.dtype)
        w2 = params["w2"].to(states.device, states.dtype)
        b2 = params["b2"].to(states.device, states.dtype)
        h = torch.einsum("bnd,bhd->bnh", states, w1) + b1.unsqueeze(-2)
        if self.activation == "tanh":
            h = torch.tanh(h)
        elif self.activation == "relu":
            h = torch.relu(h)
        elif self.activation in ("gelu",):
            h = torch.nn.functional.gelu(h)
        else:
            raise ValueError(f"unsupported MLP prior activation {self.activation!r}")
        out = torch.einsum("bnh,boh->bno", h, w2) + b2.unsqueeze(-2)
        out = out.squeeze(-1)
        if self.output_clip is not None:
            out = out.clamp(-self.output_clip, self.output_clip)
        return out

    def dones(self, states: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        return torch.zeros(states.shape[:-1], dtype=torch.bool, device=states.device)

    def evaluate(self, states: torch.Tensor, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rewards(states, params), self.dones(states, params)


# ---------------------------------------------------------------------------
# Prior sampler (mixture of the three classes)
# ---------------------------------------------------------------------------


class PriorSampler:
    """Samples batches of random reward functions from the FRE prior mixture.

    The sampler owns the three function classes and draws, for every batch
    element, a class index from the (normalised) mixture probabilities.  It then
    samples the function parameters, the encoder states (``K = 32``), the
    decoder states (``K' = 8``) and evaluates the rewards / done masks.

    For goal-reaching function rows, the HER goal is additionally *forced* into
    one of the encoder samples ("We ensure that at least one of the samples
    contains the goal state during the encoding process", Appendix B).
    """

    def __init__(
        self,
        state_dim: int,
        ratios: Any = None,
        goal_threshold: float = 0.5,
        goal_distance_type: str = "euclidean",
        goal_scale: Optional[torch.Tensor] = None,
        goal_distance_dims: Optional[Sequence[int]] = None,
        her_probs: Tuple[float, float, float] = (0.2, 0.5, 0.3),
        exclude_dims: Sequence[int] = (),
        linear_mask_zero_prob: float = 0.9,
        linear_clip_rewards: Optional[float] = None,
        mlp_hidden_dim: int = 32,
        mlp_output_clip: float = 1.0,
        mlp_scale_mode: str = "inverse_sqrt",
        num_encoder_samples: int = 32,
        num_decoder_samples: int = 8,
        batch_size: int = 512,
        device: Any = "cpu",
        seed: Optional[int] = None,
        force_goal_in_encoder: bool = True,
    ) -> None:
        self.state_dim = int(state_dim)
        self.names, self.probabilities = _normalize_ratios(ratios)
        self.num_encoder_samples = int(num_encoder_samples)
        self.num_decoder_samples = int(num_decoder_samples)
        self.batch_size = int(batch_size)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # torch generator for deterministic MLP parameter sampling
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self._global_step = 0

        self.goal_prior = GoalReachingPrior(
            state_dim=self.state_dim,
            threshold=goal_threshold,
            scale=goal_scale,
            distance_dims=goal_distance_dims,
            distance_type=goal_distance_type,
            her_probs=her_probs,
            force_goal_in_encoder=force_goal_in_encoder,
        )
        self.linear_prior = LinearPrior(
            state_dim=self.state_dim,
            exclude_dims=exclude_dims,
            mask_zero_prob=linear_mask_zero_prob,
            clip_rewards=linear_clip_rewards,
        )
        self.mlp_prior = MLPPrior(
            state_dim=self.state_dim,
            hidden_dim=mlp_hidden_dim,
            output_clip=mlp_output_clip,
            scale_mode=mlp_scale_mode,
        )
        self.force_goal_in_encoder = bool(force_goal_in_encoder)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(cls, config: Any, state_dim: int, **overrides: Any) -> "PriorSampler":
        """Build a sampler from a :class:`fre.config.default.Config`-like object."""
        ratios = getattr(config, "prior_ratios", None)
        if ratios is None:
            mixture = getattr(config, "prior_mixture", None)
            ratios = mixture if mixture is not None else None
        if isinstance(ratios, dict) and "prior_names" in ratios:  # defensive
            ratios = ratios["prior_names"]

        exclude = getattr(config, "linear_exclude_dims", ())
        if isinstance(exclude, dict):
            domain = getattr(config, "domain", None)
            exclude = exclude.get(domain, ())
        domain = getattr(config, "domain", None)
        if exclude is None:
            exclude = ()
        # AntMaze: remove XY positions from linear functions (Appendix B)
        if not exclude and domain is not None and str(domain).startswith("ant"):
            exclude = (0, 1)

        goal_distance_dims = overrides.pop("goal_distance_dims", None)
        if goal_distance_dims is None:
            goal_distance_dims = getattr(config, "goal_distance_dims", None)

        kwargs: Dict[str, Any] = dict(
            state_dim=state_dim,
            ratios=ratios,
            goal_threshold=getattr(config, "prior_goal_threshold", 0.5),
            goal_distance_type=getattr(config, "prior_goal_distance_type", "euclidean"),
            her_probs=(
                getattr(config, "her_current_prob", 0.2),
                getattr(config, "her_future_prob", 0.5),
                getattr(config, "her_random_prob", 0.3),
            ),
            exclude_dims=tuple(exclude),
            linear_mask_zero_prob=getattr(config, "linear_mask_zero_prob", 0.9),
            linear_clip_rewards=getattr(config, "linear_reward_clip", None),
            mlp_hidden_dim=getattr(config, "mlp_hidden_dim", 32),
            mlp_output_clip=getattr(config, "mlp_output_clip", 1.0),
            mlp_scale_mode=getattr(config, "mlp_scale_mode", "inverse_sqrt"),
            num_encoder_samples=getattr(config, "num_encoder_samples", 32),
            num_decoder_samples=getattr(config, "num_decoder_samples", 8),
            batch_size=getattr(config, "batch_size", 512),
            device=getattr(config, "device", "cpu"),
            seed=getattr(config, "seed", None),
            force_goal_in_encoder=getattr(config, "prior_force_goal_in_encoder", True),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # -- introspection ----------------------------------------------------
    def mixture(self) -> Tuple[List[str], List[float]]:
        """Return the (normalised) mixture as ``(names, probabilities)``."""
        return list(self.names), list(self.probabilities)

    @property
    def prior_classes(self) -> Dict[str, Any]:
        return {"goal": self.goal_prior, "linear": self.linear_prior, "mlp": self.mlp_prior}

    def set_ratios(self, ratios: Any) -> None:
        """Re-configure the mixture (used by the Table 4 prior ablations)."""
        self.names, self.probabilities = _normalize_ratios(ratios)

    def _class_for(self, name: str) -> Any:
        return self.prior_classes[name]

    # -- sampling ---------------------------------------------------------
    def sample_batch(
        self,
        dataset: Any = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        encoder_states: Optional[torch.Tensor] = None,
        decoder_states: Optional[torch.Tensor] = None,
        function_types: Optional[torch.Tensor] = None,
    ) -> PriorBatch:
        """Sample a batch of reward functions, states, rewards and done masks.

        Parameters
        ----------
        dataset: object providing ``sample_states(n)`` and (optionally)
            ``sample_trajectories(n)`` -- see :mod:`fre.rl.replay_buffer`.
            Required for the goal-reaching class (HER goal sampling).
        batch_size: number of reward functions (defaults to the configured
            ``batch_size``, i.e. 512).
        encoder_states / decoder_states: optionally pre-sampled states; the
            goal-reaching rows will still have one encoder sample overwritten by
            the goal state.
        function_types: optionally force the class index of every row (useful for
            ablations/tests).
        """
        device = torch.device(device) if device is not None else self.device
        B = int(batch_size if batch_size is not None else self.batch_size)
        K = self.num_encoder_samples
        Kp = self.num_decoder_samples
        D = self.state_dim

        params_all: Dict[str, Any] = {}
        enc_r = torch.zeros(B, K, device=device, dtype=torch.float32)
        dec_r = torch.zeros(B, Kp, device=device, dtype=torch.float32)
        enc_d = torch.zeros(B, K, device=device, dtype=torch.bool)
        dec_d = torch.zeros(B, Kp, device=device, dtype=torch.bool)

        # decide the per-row function class
        if function_types is not None:
            types = torch.as_tensor(function_types, device=device, dtype=torch.long).reshape(B)
        else:
            counts = self.rng.multinomial(B, self.probabilities)
            chunks = []
            for ci, cnt in enumerate(counts):
                if cnt:
                    chunks.append(torch.full((int(cnt),), ci, dtype=torch.long))
            types = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.long)
            perm = torch.as_tensor(self.rng.permutation(B))
            types = types[perm].to(device)
        function_names = [PRIOR_CLASS_NAMES[int(t)] for t in types.tolist()]

        # pre-sample all states (goal rows may overwrite one encoder sample each)
        all_enc = encoder_states if encoder_states is not None else self._sample_states(dataset, B * K, device, shape=(B, K, D))
        all_dec = decoder_states if decoder_states is not None else self._sample_states(dataset, B * Kp, device, shape=(B, Kp, D))
        all_enc = _to_tensor(all_enc, device).reshape(B, K, -1)
        all_dec = _to_tensor(all_dec, device).reshape(B, Kp, -1)
        all_enc = self._fit_state_dim(all_enc, device)
        all_dec = self._fit_state_dim(all_dec, device)

        # per-class evaluation
        for ci, name in enumerate(PRIOR_CLASS_NAMES):
            rows = (types == ci).nonzero(as_tuple=False).reshape(-1)
            if rows.numel() == 0:
                continue
            prior = self._class_for(name)
            n = int(rows.numel())
            if name == "goal" and dataset is None and encoder_states is None:
                raise ValueError(
                    "sampling goal-reaching rewards requires a dataset (HER goal sampling); "
                    "pass `dataset=...` or provide `encoder_states`/`decoder_states` and use "
                    "PriorSampler.sample_rewards(...)"
                )
            if name == "goal" and dataset is None:
                params = self._goal_params_from_states(all_enc[rows], all_dec[rows], device, n)
            else:
                params = prior.sample_params(
                    n, dataset=dataset, device=device, rng=self.rng, generator=self.generator
                )
            # store (detached) parameters for logging / reuse
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    params_all.setdefault(key, torch.zeros(B, *value.shape[1:], device=device, dtype=value.dtype))
                    params_all[key][rows] = value.detach()

            enc_states_rows = all_enc[rows]
            dec_states_rows = all_dec[rows]
            if name == "goal" and self.force_goal_in_encoder and K > 0:
                goal = params["goal"].unsqueeze(1).to(enc_states_rows.dtype)
                j = torch.as_tensor(self.rng.integers(0, K, size=n), device=device, dtype=torch.long)
                enc_states_rows = enc_states_rows.clone()
                enc_states_rows.scatter_(1, j.view(-1, 1, 1).expand(n, 1, D), goal)
                # default goal params if not built from states (dataset path)
                if "goal" in params:
                    pass
            rewards_e, dones_e = prior.evaluate(enc_states_rows, params)
            rewards_d, dones_d = prior.evaluate(dec_states_rows, params)
            enc_r[rows] = rewards_e.to(enc_r.dtype)
            dec_r[rows] = rewards_d.to(dec_r.dtype)
            enc_d[rows] = dones_e.to(torch.bool)
            dec_d[rows] = dones_d.to(torch.bool)
            all_enc[rows] = enc_states_rows

        self._global_step += 1
        return PriorBatch(
            encoder_states=all_enc.detach(),
            encoder_rewards=enc_r.detach(),
            encoder_dones=enc_d.detach(),
            decoder_states=all_dec.detach(),
            decoder_rewards=dec_r.detach(),
            decoder_dones=dec_d.detach(),
            function_types=types.detach(),
            function_names=function_names,
            params=params_all,
        )

    def sample_rewards(
        self,
        encoder_states: torch.Tensor,
        decoder_states: torch.Tensor,
        dataset: Any = None,
        function_types: Optional[torch.Tensor] = None,
    ) -> PriorBatch:
        """Evaluate randomly sampled reward functions on *given* states.

        Equivalent to :meth:`sample_batch` but with externally supplied states.
        When ``dataset`` is ``None`` the HER goals are drawn from the supplied
        state pool (uniform / "random state" branch of the HER distribution).
        """
        return self.sample_batch(
            dataset=dataset,
            batch_size=int(encoder_states.shape[0]),
            encoder_states=encoder_states,
            decoder_states=decoder_states,
            function_types=function_types,
        )

    def evaluate(
        self,
        encoder_states: torch.Tensor,
        decoder_states: torch.Tensor,
        function_types: Optional[torch.Tensor] = None,
        dataset: Any = None,
    ) -> PriorBatch:
        """Alias of :meth:`sample_rewards` (explicit naming for eval scripts)."""
        return self.sample_rewards(encoder_states, decoder_states, dataset=dataset, function_types=function_types)

    # -- internals --------------------------------------------------------
    def _sample_states(
        self,
        dataset: Any,
        num: int,
        device: torch.device,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Sample states from the dataset, returning ``shape``-shaped output."""
        states = _sample_states(dataset, num, device)
        states = self._fit_state_dim(states, device)
        if shape is not None:
            states = states.reshape(*shape)
        return states

    def _fit_state_dim(self, states: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Match the last dimension to ``state_dim`` (pad / truncate)."""
        d = int(states.shape[-1])
        if d == self.state_dim:
            return states
        if d > self.state_dim:
            return states[..., : self.state_dim]
        pad = torch.zeros(*states.shape[:-1], self.state_dim - d, device=device, dtype=states.dtype)
        return torch.cat([states, pad], dim=-1)

    def _goal_params_from_states(
        self,
        enc_states: torch.Tensor,
        dec_states: torch.Tensor,
        device: torch.device,
        n: int,
    ) -> Dict[str, torch.Tensor]:
        """Fallback HER-free goal sampling from a pool of provided states."""
        pool = torch.cat([enc_states.reshape(-1, self.state_dim), dec_states.reshape(-1, self.state_dim)], dim=0)
        idx = torch.as_tensor(self.rng.integers(0, pool.shape[0], size=n), device=device)
        goals = pool[idx].clone()
        u = self.rng.random(n)
        p = self.goal_prior.her_probs
        use_enc = u < (p[0] + p[1])
        if use_enc.any():
            sel = torch.as_tensor(use_enc, device=device)
            enc_pool = enc_states.reshape(n, -1, self.state_dim)
            pick = torch.as_tensor(
                self.rng.integers(0, enc_pool.shape[1], size=n), device=device, dtype=torch.long
            )
            candidate = enc_pool.gather(1, pick.view(n, 1, 1).expand(n, 1, self.state_dim)).squeeze(1)
            goals = torch.where(sel.view(-1, 1), candidate, goals)
        return {"goal": goals.detach(), "her_current_states": enc_states[:, 0].detach()}


def make_prior_sampler(config: Any, state_dim: int, **overrides: Any) -> PriorSampler:
    """Module-level convenience wrapper around :meth:`PriorSampler.from_config`."""
    return PriorSampler.from_config(config, state_dim, **overrides)


# ---------------------------------------------------------------------------
# Minimal dataset used for unit tests / datasets without trajectory sampling
# ---------------------------------------------------------------------------


class ArrayPriorDataset:
    """Lightweight in-memory dataset for the prior sampler.

    Wraps an array of states ``(N, state_dim)`` and an optional trajectory
    length ``T``.  When ``T > 1`` the states are interpreted as ``N / T``
    consecutive trajectories of length ``T`` so that the HER "future state"
    branch is meaningful.  This mirrors the interface expected from
    :class:`fre.rl.replay_buffer.ReplayBuffer`.
    """

    def __init__(
        self,
        states: Any,
        trajectory_length: Optional[int] = None,
        terminals: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        arr = states if isinstance(states, np.ndarray) else np.asarray(states, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"states must be 2-D (N, D), got shape {arr.shape}")
        self.states = arr.astype(np.float32)
        self.state_dim = int(arr.shape[-1])
        self.trajectory_length = int(trajectory_length) if trajectory_length else None
        self.terminals = None if terminals is None else np.asarray(terminals)
        self.rng = np.random.default_rng(seed)
        std = self.states.std(axis=0)
        self.state_std = np.maximum(std, 1e-6).astype(np.float32)

    # sampling ------------------------------------------------------------
    def sample_states(self, num: int) -> np.ndarray:
        idx = self.rng.integers(0, self.states.shape[0], size=int(num))
        return self.states[idx]

    def sample_trajectories(self, num: int) -> Tuple[np.ndarray, np.ndarray]:
        num = int(num)
        if not self.trajectory_length or self.trajectory_length <= 1:
            states = self.sample_states(num).reshape(num, 1, self.state_dim)
            return states, np.ones((num, 1), dtype=np.float32)
        T = self.trajectory_length
        n_traj = self.states.shape[0] // T
        if n_traj <= 0:
            states = self.states.reshape(1, -1, self.state_dim)[:, :T]
            mask = np.ones(states.shape[:2], dtype=np.float32)
            return states, mask
        idx = self.rng.integers(0, n_traj, size=num)
        states = self.states.reshape(n_traj, T, self.state_dim)[idx]
        mask = np.ones((num, T), dtype=np.float32)
        if self.terminals is not None:
            term = self.terminals.reshape(n_traj, T)[idx]
            mask = (1.0 - (term > 0.5).astype(np.float32))
        return states.astype(np.float32), mask

    def __len__(self) -> int:
        return int(self.states.shape[0])
