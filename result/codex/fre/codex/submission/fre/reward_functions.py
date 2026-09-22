"""Prior distributions over random unsupervised reward functions.

Section 4.2 of the paper ("Random Functions as a Prior Reward Distribution")
defines a mixture of three families of random reward functions, each with
progressively higher complexity:

  1. singleton goal-reaching rewards  (reward -1 until a goal is reached, 0 after)
  2. random linear functions          (inner product with a sparse random vector)
  3. random MLPs                      (2-layer MLP with random weights, tanh)

Appendix B gives the exact generation procedure, which is implemented here:

  * Goal-reaching rewards use a hindsight-experience-relabelling (HER)
    distribution over goals.  For a randomly selected state, the goal is that
    state with probability 0.2, a future state in the same trajectory with
    probability 0.5, and a completely random dataset state with probability 0.3.
  * Random linear functions use a uniform vector in [-1, 1] with an independent
    Bernoulli(0.9) mask zeroing each dimension (bias towards sparse / simple
    functions).  On AntMaze the (x, y) position dimensions are excluded from
    linear functions, since their scale caused training instability.
  * Random MLPs have hidden size 32, parameters drawn from a normal
    distribution scaled by the average dimension of the layer, a tanh
    activation, and the output is clipped to [-1, 1].

The paper evaluates several *subsets* of these families (Sections 5.3 and 5.4):
FRE-all (equal mixture), FRE-goals, FRE-lin, FRE-mlp, FRE-lin-mlp,
FRE-goal-mlp, FRE-goal-lin.  All of these are expressible with
``MixturePrior`` by changing the mixture ratios (see ``build_prior``).

Additionally, Section 5.4 / Figure 6 augments the prior with domain-specific
reward families (``FRE-hint``): unit-direction movement rewards for
ant-directional and specific-velocity rewards for the ExORL velocity tasks.
Those families live in ``fre.tasks`` because they are domain specific, and are
exposed through ``build_prior`` via the ``hint_tasks`` argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------------------
# Reward discretisation (used by the encoder input pipeline)
# --------------------------------------------------------------------------------------
def discretize_reward(
    rewards: torch.Tensor,
    num_bins: int = 32,
    r_min: float = -1.0,
    r_max: float = 1.0,
) -> torch.Tensor:
    """Discretise a scalar reward into ``num_bins`` bins.

    The addendum specifies the exact procedure used by FRE: "the scalar reward
    is discretized into 32 bins by rescaling the reward to [0, 1] and then
    multiplying by 32 and flooring to the nearest integer".

    Each reward family has a known output range (``[-1, 0]`` for goal-reaching
    rewards, ``[-1, 1]`` for linear / MLP rewards), so the rescaling is done
    with a fixed analytic range rather than an empirical one.  This keeps the
    discretisation stationary across training iterations, which matters because
    the bin index is looked up in a learned embedding table.

    Returns an int64 tensor of bin indices in ``[0, num_bins - 1]``.
    """
    if r_max <= r_min:
        raise ValueError("r_max must be greater than r_min")
    scaled = (rewards - r_min) / (r_max - r_min)
    scaled = scaled.clamp(0.0, 1.0)
    bins = torch.floor(scaled * num_bins)
    return bins.clamp_(0, num_bins - 1).long()


# --------------------------------------------------------------------------------------
# Batched reward functions
# --------------------------------------------------------------------------------------
def _align_bias(bias: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a ``(B, H)`` bias so it broadcasts against a ``(B, ..., H)`` tensor."""
    while bias.dim() < ndim:
        bias = bias.unsqueeze(-2)
    return bias


class BatchedRewardFunctions:
    """A batch of ``B`` reward functions that can be evaluated on batches of states.

    ``reward(states)`` accepts a tensor of shape ``(B, ..., state_dim)`` and
    returns a tensor of shape ``(B, ...)``.  Every family contributes its own
    analytic reward range (``reward_range``), which is used by the encoder to
    discretise the reward into bins.
    """

    # Analytic output range of the reward family, per batch element.
    r_min: torch.Tensor  # (B,)
    r_max: torch.Tensor  # (B,)

    def reward(self, states: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def maybe_insert_goal_state(self, states: torch.Tensor) -> torch.Tensor:
        """Hook used by goal-reaching priors.

        Appendix B: "We ensure that at least one of the samples contains the
        goal state during the encoding process."  Goal-reaching priors replace
        the last encoding state of each row with the goal; all other families
        are unchanged.
        """
        return states

    def done(self, states: torch.Tensor) -> torch.Tensor:
        """Task-termination mask for a batch of states.

        Appendix B: for random goal-reaching reward functions "a done mask is
        set to True when the goal is achieved".  The mask is combined with the
        dataset's own terminal flags in the Bellman backup, so that reaching the
        goal stops bootstrapping (important because the goal-reaching reward is
        ``0`` from that point onwards).
        """
        return torch.zeros(states.shape[:-1], device=states.device, dtype=torch.float32)


class GoalReachingBatchedRewards(BatchedRewardFunctions):
    """Singleton goal-reaching rewards: ``-1`` until the goal is reached, ``0`` after."""

    def __init__(
        self,
        goals: torch.Tensor,
        threshold: float,
        dims: Optional[Sequence[int]] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> None:
        self.goals = goals  # (B, D)
        self.threshold = float(threshold)
        # If ``dims`` is given, the goal distance only considers these state
        # dimensions (AntMaze uses the (x, y) position, distance threshold 2).
        self.dims = None if dims is None else list(dims)
        # ``scale`` optionally normalises each state dimension before the
        # distance is computed (ExORL goal-reaching uses the standard deviation
        # of every observation dimension over the offline dataset).
        self.scale = None if scale is None else torch.as_tensor(scale, dtype=torch.float32)
        b = goals.shape[0]
        self.r_min = torch.full((b,), -1.0, device=goals.device)
        self.r_max = torch.zeros(b, device=goals.device)

    def _distance(self, states: torch.Tensor) -> torch.Tensor:
        # ``states`` is either ``(B, K, D)`` (an encoding / decoding set) or
        # ``(B, D)`` (a flat batch of transitions).
        scale = None
        if self.scale is not None:
            scale = self.scale.to(states.device)
        if states.dim() == 2:
            if self.dims is None:
                delta = states - self.goals
                if scale is not None:
                    delta = delta / scale
            else:
                s = states[..., self.dims]
                g = self.goals[..., self.dims]
                delta = s - g
                if scale is not None:
                    delta = delta / scale[..., self.dims]
        elif self.dims is None:
            delta = states - self.goals.unsqueeze(1)
            if scale is not None:
                delta = delta / scale
        else:
            delta = states[..., self.dims] - self.goals[:, self.dims].unsqueeze(1)
            if scale is not None:
                delta = delta / scale[..., self.dims]
        return torch.linalg.vector_norm(delta, dim=-1)

    def reward(self, states: torch.Tensor) -> torch.Tensor:
        dist = self._distance(states)
        return torch.where(dist < self.threshold, torch.zeros_like(dist), -torch.ones_like(dist))

    def maybe_insert_goal_state(self, states: torch.Tensor) -> torch.Tensor:
        # Put the goal state into the last slot of every encoding set.
        out = states.clone()
        out[:, -1] = self.goals
        return out

    def done(self, states: torch.Tensor) -> torch.Tensor:
        dist = self._distance(states)
        return (dist < self.threshold).to(torch.float32)


class LinearBatchedRewards(BatchedRewardFunctions):
    """Random linear rewards: ``eta(s) = <w, s>`` with a sparse random ``w``."""

    def __init__(self, weights: torch.Tensor, clip: float = 1.0) -> None:
        self.weights = weights  # (B, D)
        self.clip = float(clip)
        b = weights.shape[0]
        self.r_min = torch.full((b,), -clip, device=weights.device)
        self.r_max = torch.full((b,), clip, device=weights.device)

    def reward(self, states: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("b...d,bd->b...", states, self.weights)
        if self.clip is not None:
            out = out.clamp(-self.clip, self.clip)
        return out


class MLPBatchedRewards(BatchedRewardFunctions):
    """Random 2-layer MLP rewards with a tanh nonlinearity, output clipped to [-1, 1]."""

    def __init__(
        self,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        clip: float = 1.0,
    ) -> None:
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        self.clip = float(clip)
        b = w1.shape[0]
        self.r_min = torch.full((b,), -clip, device=w1.device)
        self.r_max = torch.full((b,), clip, device=w1.device)

    def reward(self, states: torch.Tensor) -> torch.Tensor:
        # Flatten any leading "set" dimensions to a single axis, apply the MLP
        # with batched matrix products, then restore the original shape.
        lead = states.shape[1:-1]
        flat = states.reshape(states.shape[0], -1, states.shape[-1])
        # Broadcast a single reward function (batch of one) across the state batch.
        n = flat.shape[0]
        if self.w1.shape[0] == 1 and n != 1:
            w1 = self.w1.expand(n, -1, -1)
            b1 = self.b1.expand(n, -1)
            w2 = self.w2.expand(n, -1, -1)
            b2 = self.b2.expand(n)
        else:
            w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        h = torch.tanh(torch.bmm(flat, w1) + b1.unsqueeze(1))
        out = torch.bmm(h, w2).squeeze(-1) + b2.unsqueeze(1)
        out = out.clamp(-self.clip, self.clip)
        return out.reshape(states.shape[0], *lead)


class MixtureBatchedRewards(BatchedRewardFunctions):
    """Selects, per batch element, one of several batched reward families."""

    def __init__(self, members: List[BatchedRewardFunctions], assignment: torch.Tensor) -> None:
        self.members = members
        self.assignment = assignment  # (B,) long, index into ``members``
        self.r_min = torch.stack([m.r_min for m in members], dim=0)
        self.r_max = torch.stack([m.r_max for m in members], dim=0)
        # Per-element ranges chosen according to the assignment.
        self.r_min = self.r_min.gather(0, assignment.view(1, -1)).squeeze(0)
        self.r_max = self.r_max.gather(0, assignment.view(1, -1)).squeeze(0)

    def reward(self, states: torch.Tensor) -> torch.Tensor:
        assignment = self._assignment_for(states.shape[0])
        rewards = torch.stack([m.reward(states) for m in self.members], dim=0)  # (M, B, ...)
        idx = assignment.view(-1, *([1] * (rewards.dim() - 2)))
        idx = idx.expand(-1, *rewards.shape[2:])
        return rewards.gather(0, idx.unsqueeze(0)).squeeze(0)

    def maybe_insert_goal_state(self, states: torch.Tensor) -> torch.Tensor:
        assignment = self._assignment_for(states.shape[0])
        out = states
        for i, member in enumerate(self.members):
            mask = (assignment == i)
            if mask.any():
                updated = member.maybe_insert_goal_state(states)
                out = torch.where(mask.view(-1, *([1] * (states.dim() - 1))), updated, out)
        return out

    def _assignment_for(self, batch_size: int) -> torch.Tensor:
        """Broadcast a single reward function across a larger batch of states."""
        if self.assignment.shape[0] == batch_size:
            return self.assignment
        if self.assignment.shape[0] == 1:
            return self.assignment.expand(batch_size)
        raise ValueError(
            f"reward function batch ({self.assignment.shape[0]}) does not match "
            f"state batch ({batch_size})"
        )

    def done(self, states: torch.Tensor) -> torch.Tensor:
        assignment = self._assignment_for(states.shape[0])
        masks = torch.stack([m.done(states) for m in self.members], dim=0)  # (M, B, ...)
        idx = assignment.view(-1, *([1] * (masks.dim() - 2)))
        idx = idx.expand(-1, *masks.shape[2:])
        return masks.gather(0, idx.unsqueeze(0)).squeeze(0)


# --------------------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------------------
class RewardPrior:
    """Base class for prior distributions over reward functions.

    Sub-classes implement :meth:`sample`, returning a
    :class:`BatchedRewardFunctions` object of batch size ``batch_size``.
    """

    def sample(self, batch_size: int, device: torch.device) -> BatchedRewardFunctions:
        raise NotImplementedError


class GoalReachingPrior(RewardPrior):
    """Singleton goal-reaching rewards with goals sampled via a HER distribution."""

    def __init__(
        self,
        goal_sampler: Callable[[int], torch.Tensor],
        threshold: float = 2.0,
        dims: Optional[Sequence[int]] = None,
        scale: Optional[Sequence[float]] = None,
    ) -> None:
        self.goal_sampler = goal_sampler
        self.threshold = float(threshold)
        self.dims = None if dims is None else list(dims)
        self.scale = None if scale is None else torch.as_tensor(scale, dtype=torch.float32)

    def sample(self, batch_size: int, device: torch.device) -> BatchedRewardFunctions:
        goals = self.goal_sampler(batch_size).to(device)
        return GoalReachingBatchedRewards(goals, self.threshold, self.dims, self.scale)


class LinearPrior(RewardPrior):
    """Random linear rewards with a Bernoulli(0.9) sparsity mask."""

    def __init__(
        self,
        state_dim: int,
        mask_prob: float = 0.9,
        excluded_dims: Optional[Sequence[int]] = None,
        clip: float = 1.0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.mask_prob = float(mask_prob)
        self.excluded_dims = None if excluded_dims is None else list(excluded_dims)
        self.clip = float(clip)

    def sample(self, batch_size: int, device: torch.device) -> BatchedRewardFunctions:
        w = torch.empty(batch_size, self.state_dim, device=device).uniform_(-1.0, 1.0)
        if self.excluded_dims:
            w[:, self.excluded_dims] = 0.0
        keep = (torch.rand(batch_size, self.state_dim, device=device) >= self.mask_prob).float()
        return LinearBatchedRewards(w * keep, clip=self.clip)


class MLPPrior(RewardPrior):
    """Random 2-layer MLP rewards (hidden size 32, tanh, clipped output)."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 32,
        clip: float = 1.0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.clip = float(clip)

    def sample(self, batch_size: int, device: torch.device) -> BatchedRewardFunctions:
        d, h = self.state_dim, self.hidden_dim
        # "Parameters are sampled using a normal distribution scaled by the
        # average dimension of the layer."
        std1 = 1.0 / math.sqrt(0.5 * (d + h))
        std2 = 1.0 / math.sqrt(0.5 * (h + 1))
        w1 = torch.randn(batch_size, d, h, device=device) * std1
        b1 = torch.randn(batch_size, h, device=device) * std1
        w2 = torch.randn(batch_size, h, 1, device=device) * std2
        b2 = torch.randn(batch_size, device=device) * std2
        return MLPBatchedRewards(w1, b1, w2, b2, clip=self.clip)


class MixturePrior(RewardPrior):
    """A mixture of reward families, with per-family ratios.

    ``ratios`` are normalised to sum to one.  A ratio of zero disables a
    family entirely (used by the Section 5.3 ablation over subsets of families).
    """

    def __init__(self, members: Dict[str, RewardPrior], ratios: Dict[str, float]) -> None:
        self.members = members
        self.names = [n for n in ratios if ratios[n] > 0 and n in members]
        total = sum(ratios[n] for n in self.names)
        if total <= 0:
            raise ValueError("MixturePrior needs at least one family with a positive ratio")
        self.probs = torch.tensor([ratios[n] / total for n in self.names], dtype=torch.float32)

    def sample(self, batch_size: int, device: torch.device) -> BatchedRewardFunctions:
        probs = self.probs.to(device)
        assignment = torch.multinomial(probs, batch_size, replacement=True)
        member_rewards = []
        for name in self.names:
            member_rewards.append(self.members[name].sample(batch_size, device))
        if len(member_rewards) == 1:
            return member_rewards[0]
        return MixtureBatchedRewards(member_rewards, assignment)


# --------------------------------------------------------------------------------------
# Registry / factory
# --------------------------------------------------------------------------------------
def _normalise_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in ratios.values())
    if total <= 0:
        raise ValueError("ratios must contain at least one positive entry")
    return {k: max(0.0, float(v)) / total for k, v in ratios.items()}


# Canonical ablation configurations from Sections 5.2 - 5.4.
PRIOR_REGISTRY: Dict[str, Dict[str, float]] = {
    "FRE-all": {"goal": 1 / 3, "lin": 1 / 3, "mlp": 1 / 3},
    "FRE-goals": {"goal": 1.0, "lin": 0.0, "mlp": 0.0},
    "FRE-lin": {"goal": 0.0, "lin": 1.0, "mlp": 0.0},
    "FRE-mlp": {"goal": 0.0, "lin": 0.0, "mlp": 1.0},
    "FRE-lin-mlp": {"goal": 0.0, "lin": 0.5, "mlp": 0.5},
    "FRE-goal-mlp": {"goal": 0.5, "lin": 0.0, "mlp": 0.5},
    "FRE-goal-lin": {"goal": 0.5, "lin": 0.5, "mlp": 0.0},
}


def build_prior(
    name: str,
    state_dim: int,
    goal_sampler: Callable[[int], torch.Tensor],
    *,
    goal_threshold: float = 2.0,
    goal_dims: Optional[Sequence[int]] = None,
    goal_scale: Optional[Sequence[float]] = None,
    linear_excluded_dims: Optional[Sequence[int]] = None,
    ratios: Optional[Dict[str, float]] = None,
    hint_priors: Optional[Dict[str, RewardPrior]] = None,
    hint_ratios: Optional[Dict[str, float]] = None,
) -> RewardPrior:
    """Build a prior reward distribution by name.

    ``name`` is one of the keys of :data:`PRIOR_REGISTRY`, or ``"FRE-hint"``
    (Sections 5.4 / Figure 6) in which case ``hint_priors`` supplies the
    domain-specific reward families.
    """
    if name == "FRE-hint":
        if not hint_priors:
            raise ValueError("FRE-hint requires hint_priors")
        members: Dict[str, RewardPrior] = dict(hint_priors)
        member_ratios = dict(hint_ratios or {k: 1.0 / len(members) for k in members})
        # The domain-specific families replace part of the generic FRE-all
        # mixture; any generic family with a positive ratio is added back in.
        generic = {
            "goal": lambda: GoalReachingPrior(goal_sampler, goal_threshold, goal_dims, goal_scale),
            "lin": lambda: LinearPrior(state_dim, excluded_dims=linear_excluded_dims),
            "mlp": lambda: MLPPrior(state_dim),
        }
        for key, factory in generic.items():
            if member_ratios.get(key, 0.0) > 0.0:
                members.setdefault(key, factory())
        return MixturePrior(members, member_ratios)

    if name not in PRIOR_REGISTRY:
        raise ValueError(f"Unknown prior '{name}'. Options: {sorted(PRIOR_REGISTRY)} or 'FRE-hint'")

    member_ratios = _normalise_ratios(ratios or PRIOR_REGISTRY[name])
    members = {
        "goal": GoalReachingPrior(goal_sampler, goal_threshold, goal_dims, goal_scale),
        "lin": LinearPrior(state_dim, excluded_dims=linear_excluded_dims),
        "mlp": MLPPrior(state_dim),
    }
    return MixturePrior(members, member_ratios)


@dataclass
class PriorConfig:
    """Serialisable description of a prior reward distribution."""

    name: str = "FRE-all"
    ratios: Optional[Dict[str, float]] = None
    goal_threshold: float = 2.0
    goal_dims: Optional[List[int]] = None
    linear_excluded_dims: Optional[List[int]] = None
    hint_tasks: Optional[List[str]] = field(default=None)

    def build(self, state_dim: int, goal_sampler, hint_priors=None) -> RewardPrior:
        return build_prior(
            self.name,
            state_dim,
            goal_sampler,
            goal_threshold=self.goal_threshold,
            goal_dims=self.goal_dims,
            linear_excluded_dims=self.linear_excluded_dims,
            ratios=self.ratios,
            hint_priors=hint_priors,
        )
