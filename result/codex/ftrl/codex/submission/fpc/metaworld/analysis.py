"""Representation-level analyses for the robotic manipulation experiments.

* :func:`collect_expert_loglikelihoods` -- log-likelihood of expert state-action
  pairs ``(s, a*), a* ~ pi_*(s)`` under the fine-tuned policy, together with the
  2D PCA projection of the states (Figure 8).  The likelihoods are computed
  every 50K training steps (addendum).
* :func:`collect_cka_curve` -- CKA between the pre-training activations and the
  fine-tuned activations, layer by layer (Figure 20).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..analysis.cka import cka_over_training, layer_activations
from ..analysis.forgetting import expert_log_likelihood, pca_projection


def collect_expert_trajectories(teacher, config, stages, num_samples: int = 5000, seed: int = 0):
    """Collect ``(state, a*)`` pairs with the expert/pre-trained policy."""

    from .robotic_sequence import RoboticSequence

    env = RoboticSequence(config, seed=seed, stages=list(stages))
    states, actions = [], []
    obs, _ = env.reset(seed=seed)
    stage = 0
    while len(states) < num_samples:
        action = teacher.select_action(obs, stage, deterministic=True)
        states.append(obs)
        actions.append(action)
        result = env.step(action)
        obs, stage = result.observation, result.stage
        if result.terminated or result.truncated:
            obs, _ = env.reset(seed=seed + len(states))
            stage = 0
    env.close()
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def gaussian_log_prob(agent, observations: Tensor, actions: Tensor, stage: Optional[Tensor] = None) -> Tensor:
    """``log pi(a | s)`` for the squashed Gaussian policy of :class:`SAC`."""

    if stage is None:
        stage = torch.zeros(observations.shape[0], dtype=torch.long, device=observations.device)
    _, _, mu, std = agent.actor(observations, stage, with_logprob=False)
    # invert the tanh squashing
    clipped = actions.clamp(-1 + 1e-6, 1 - 1e-6)
    pre_tanh = 0.5 * torch.log((1 + clipped) / (1 - clipped))
    dist = torch.distributions.Normal(mu, std)
    log_prob = dist.log_prob(pre_tanh) - torch.log(1 - clipped.pow(2) + 1e-6)
    return log_prob.sum(dim=-1)


def collect_expert_loglikelihoods(
    agent,
    states: np.ndarray,
    actions: np.ndarray,
    checkpoint_steps: Sequence[int],
    stage: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Return ``{"checkpoint_<step>": likelihoods, "pca_<step>": projection}``.

    The likelihoods are computed every 50K training steps (addendum).
    """

    observations = torch.as_tensor(states, dtype=torch.float32, device=device)
    expert_actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    stage_t = None if stage is None else torch.as_tensor(stage, dtype=torch.long, device=device)
    projection = pca_projection(states, n_components=2)
    out: Dict[str, np.ndarray] = {}
    for step in checkpoint_steps:
        likelihoods = expert_log_likelihood(
            lambda obs, act, s=stage_t: gaussian_log_prob(agent, obs, act, s),
            observations,
            expert_actions,
        )
        out[f"checkpoint_{step}"] = likelihoods
        out[f"pca_{step}"] = projection
    return out


def collect_cka_curve(
    agent,
    reference_activations: Dict[str, Tensor],
    probe_states: np.ndarray,
    checkpoint_steps: Sequence[int],
    stage: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, List[float]]]:
    """CKA between pre-training and current activations, per layer and step."""

    probe = torch.as_tensor(probe_states, dtype=torch.float32, device=device)
    curve: Dict[str, Dict[str, List[float]]] = {}
    steps: List[int] = []
    per_step: List[Dict[str, float]] = []
    for _ in checkpoint_steps:
        current = layer_activations(agent.actor, probe)
        per_step.append(cka_over_training(reference_activations, current))
        steps.append(0)
    for layer in per_step[0]:
        curve[layer] = {
            "steps": [int(s) for s in checkpoint_steps],
            "cka": [float(step[layer]) for step in per_step],
        }
    return curve


def collect_reference_activations(actor, probe_states: np.ndarray, device: str = "cpu") -> Dict[str, Tensor]:
    """Activations of the pre-trained actor on the probe states (Figure 20)."""

    probe = torch.as_tensor(probe_states, dtype=torch.float32, device=device)
    return layer_activations(actor, probe)
