"""AppleRetrieval -- a 1D grid-world exhibiting a state coverage gap.

Appendix A.2.  The environment has two phases:

* Phase 1:  start at ``x = 0`` and go right until ``x = M`` to retrieve an
  apple.  The observation is ``o = [-c]``.
* Phase 2:  go back to ``x = 0`` carrying the apple.  The observation is
  ``o = [c]``.

In both phases the reward is ``+1`` for stepping in the correct direction and
``-1`` otherwise.  Episodes terminate when the goal of the current phase is
reached, or after 100 timesteps.

Because Phase 1 (CLOSE) has to be completed before Phase 2 (FAR) is ever
visited, the environment directly instantiates a state coverage gap.  The policy
is a linear model ``pi_{w,b}(o) = sigmoid(w . o + b)`` and is fine-tuned with
REINFORCE starting from the parameters obtained by training on Phase 2 alone.

The observation scaling ``c`` controls whether the pre-trained solution relies
on the weight (``|w| >> |b|``, little forgetting) or on the bias
(``|b| >> |w|``, strong forgetting), matching Figure 11 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PHASE1, PHASE2 = 0, 1


class LinearPolicy(nn.Module):
    """``pi(move right | o) = sigmoid(w * o + b)`` as in Appendix A.2."""

    def __init__(self, c: float = 1.0, w: float = 0.0, b: float = 0.0) -> None:
        super().__init__()
        self.c = float(c)
        self.weight = nn.Parameter(torch.tensor(float(w)))
        self.bias = nn.Parameter(torch.tensor(float(b)))

    def forward(self, phase: int) -> torch.Tensor:
        o = -self.c if phase == PHASE1 else self.c
        logit = self.weight * o + self.bias
        return torch.sigmoid(logit)

    @property
    def wb_ratio(self) -> float:
        return float(abs(self.bias.detach()) / max(abs(self.weight.detach()), 1e-8))


@dataclass
class AppleRetrieval:
    """1D grid-world with two phases (Appendix A.2)."""

    M: int = 30
    c: float = 1.0
    max_steps: int = 100
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    def rollout(
        self,
        policy: LinearPolicy,
        phase: int = PHASE1,
        start_x: Optional[int] = None,
        greedy: bool = False,
        record: bool = False,
    ) -> Dict[str, object]:
        """Run a single episode of one phase with the given policy."""

        x = 0 if start_x is None else start_x
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        trajectory: List[int] = []
        reached_goal = False

        for _ in range(self.max_steps):
            p_right = policy(phase)
            if greedy:
                action = int(p_right.detach().item() >= 0.5)
            else:
                action = int(torch.bernoulli(p_right).item())
            if record:
                log_probs.append(torch.log(p_right if action else 1.0 - p_right + 1e-12))
            correct_right = phase == PHASE1
            reward = 1.0 if (bool(action) == correct_right) else -1.0
            rewards.append(reward)
            x += 1 if action else -1
            x = int(np.clip(x, 0, self.M))
            trajectory.append(x)
            if phase == PHASE1 and x >= self.M:
                reached_goal = True
                break
            if phase == PHASE2 and x <= 0:
                reached_goal = True
                break

        out: Dict[str, object] = {
            "reached_goal": reached_goal,
            "returns": rewards,
            "return": float(np.sum(rewards)),
            "trajectory": trajectory,
        }
        if record:
            out["log_probs"] = log_probs
        return out

    def rollout_full(
        self,
        policy: LinearPolicy,
        greedy: bool = False,
        record: bool = False,
    ) -> Dict[str, object]:
        """Roll out both phases; counts as success only if both are solved."""

        p1 = self.rollout(policy, PHASE1, greedy=greedy, record=record)
        if not p1["reached_goal"]:
            return {
                "success": False,
                "phase1": p1,
                "phase2": None,
                "return": p1["return"],
            }
        p2 = self.rollout(policy, PHASE2, start_x=self.M, greedy=greedy, record=record)
        return {
            "success": bool(p2["reached_goal"]),
            "phase1": p1,
            "phase2": p2,
            "return": p1["return"] + p2["return"],
        }

    # ------------------------------------------------------------------
    def evaluate(
        self,
        policy: LinearPolicy,
        episodes: int = 200,
        phase: Optional[int] = None,
        greedy: bool = True,
    ) -> float:
        """Success rate over ``episodes`` for the full task or a single phase."""

        successes = 0
        for _ in range(episodes):
            if phase is None:
                successes += int(self.rollout_full(policy, greedy=greedy)["success"])
            elif phase == PHASE1:
                successes += int(self.rollout(policy, PHASE1, greedy=greedy)["reached_goal"])
            else:
                successes += int(
                    self.rollout(policy, PHASE2, start_x=self.M, greedy=greedy)["reached_goal"]
                )
        return successes / episodes


def _simulate_batch(
    env: AppleRetrieval,
    p_right_1: float,
    p_right_2: float,
    num_episodes: int,
    rng: np.random.Generator,
    phase: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Vectorised simulation of ``num_episodes`` episodes.

    Because the policy only depends on the phase (the observation is constant
    within a phase), every step in a phase shares the same action probability.
    We therefore only need the *counts* of right/left actions per phase to build
    the exact REINFORCE log-likelihood of the batch.
    """

    n = num_episodes
    zeros = np.zeros(n)
    n1r, n1l = zeros.copy(), zeros.copy()
    n2r, n2l = zeros.copy(), zeros.copy()
    success = np.zeros(n, dtype=bool)

    if phase in (None, PHASE1):
        x = np.zeros(n, dtype=np.int64)
        budget = np.full(n, env.max_steps)
        active = np.ones(n, dtype=bool)
        for _ in range(env.max_steps):
            if not active.any():
                break
            draw = rng.random(n) < p_right_1
            right = draw & active
            left = (~draw) & active
            n1r += right
            n1l += left
            x = np.where(active, np.clip(x + right.astype(np.int64) - left.astype(np.int64), 0, env.M), x)
            reached = x >= env.M
            active &= ~reached
            budget = np.where(active, budget - 1, budget)
        if phase == PHASE1:
            return {"n1r": n1r, "n1l": n1l, "n2r": n2r, "n2l": n2l, "success": ~active}
        success = ~active
        phase2_budget = np.maximum(budget, 0)
    else:  # phase == PHASE2 -- only the second phase is simulated
        success = np.ones(n, dtype=bool)
        phase2_budget = np.full(n, env.max_steps)

    # Phase 2: start at x = M and walk back to 0 (left is the correct action).
    x = np.full(n, env.M, dtype=np.int64)
    active2 = (phase2_budget > 0) & success
    for _ in range(env.max_steps):
        if not active2.any():
            break
        draw_right = rng.random(n) < p_right_2
        right = draw_right & active2
        left = (~draw_right) & active2
        n2r += right
        n2l += left
        x = np.where(active2, np.clip(x + right.astype(np.int64) - left.astype(np.int64), 0, env.M), x)
        reached = x <= 0
        active2 &= ~reached
    ok2 = (~active2) & success
    if phase == PHASE2:
        return {"n1r": n1r, "n1l": n1l, "n2r": n2r, "n2l": n2l, "success": ok2}
    return {"n1r": n1r, "n1l": n1l, "n2r": n2r, "n2l": n2l, "success": ok2}


def reinforce(
    env: AppleRetrieval,
    policy: LinearPolicy,
    episodes: int = 2000,
    lr: float = 1e-2,
    phase: Optional[int] = None,
    baseline: bool = True,
    seed: Optional[int] = None,
    log_every: int = 0,
    episodes_per_update: int = 64,
) -> Dict[str, List[float]]:
    """REINFORCE (Williams, 1992) on the full task or on a single phase.

    ``phase=None`` trains on the full two-phase episode (fine-tuning), while
    ``phase=PHASE2`` trains the pre-trained solution used to initialise
    fine-tuning.

    The implementation is a *batched* REINFORCE: ``episodes_per_update``
    episodes are simulated with NumPy and a single gradient step is taken per
    batch.  Because the linear policy depends only on the phase, the
    log-likelihood of a batch is a function of the per-phase action counts, so
    the estimator is exactly the standard REINFORCE estimator.
    """

    rng = np.random.default_rng(seed)
    optimiser = torch.optim.SGD(policy.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"return": [], "success": [], "wb_ratio": []}

    num_updates = max(episodes // episodes_per_update, 1)
    for update in range(num_updates):
        with torch.no_grad():
            p1 = float(policy(PHASE1).item())
            p2 = float(policy(PHASE2).item())
        counts = _simulate_batch(env, p1, p2, episodes_per_update, rng, phase=phase)

        n1r, n1l = counts["n1r"], counts["n1l"]
        n2r, n2l = counts["n2r"], counts["n2l"]
        total_n = n1r + n1l + n2r + n2l
        # Rewards: right in phase 1 is correct (+1), right in phase 2 is wrong (-1).
        total_r = n1r - n1l - n2r + n2l
        mean_r = np.where(total_n > 0, total_r / np.maximum(total_n, 1), 0.0)

        # Sum of (r_t - baseline) over the steps of each action type.
        s1r, s1l = n1r * (1.0 - mean_r), n1l * (-1.0 - mean_r)
        s2r, s2l = n2r * (-1.0 - mean_r), n2l * (1.0 - mean_r)

        # Use the policy parameters directly so that gradients flow.
        p1_t = policy(PHASE1)
        p2_t = policy(PHASE2)
        log_p1 = torch.log(p1_t + 1e-12)
        log_q1 = torch.log(1.0 - p1_t + 1e-12)
        log_p2 = torch.log(p2_t + 1e-12)
        log_q2 = torch.log(1.0 - p2_t + 1e-12)

        to_t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        loss = -(to_t(s1r) * log_p1 + to_t(s1l) * log_q1 + to_t(s2r) * log_p2 + to_t(s2l) * log_q2)
        loss = loss.mean() / max(float(total_n.mean()), 1.0)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimiser.step()

        history["return"].append(float(total_r.mean()))
        history["success"].append(float(counts["success"].mean()))
        history["wb_ratio"].append(policy.wb_ratio)
        if log_every and (update + 1) % log_every == 0:
            print(
                f"[appleretrieval] update {update + 1:5d}  return {np.mean(history['return'][-log_every:]):7.2f}"
                f"  success {np.mean(history['success'][-log_every:]):.2f}  |b/w| {policy.wb_ratio:.2f}"
            )
    return history


def pretrain_on_phase2(
    env: AppleRetrieval,
    episodes: int = 2000,
    lr: float = 1e-2,
    seed: Optional[int] = None,
) -> Tuple[LinearPolicy, Dict[str, List[float]]]:
    """Train the pre-trained policy ``pi_*`` on Phase 2 alone."""

    policy = LinearPolicy(c=env.c, w=0.0, b=0.0)
    history = reinforce(env, policy, episodes=episodes, lr=lr, phase=PHASE2, seed=seed)
    return policy, history
