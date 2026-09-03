"""Integrated Gradients explanation baseline for RICE.

This module provides an Integrated Gradients (IG) feature-attribution scorer
that can be swapped in place of the learned mask network when selecting
critical states for refinement.  It attributes the value or the log-probability
of the executed action back to the input observation and returns a single
scalar criticality score per state.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class IntegratedGradients:
    """Integrated-Gradients criticality scorer for a target policy.

    The scorer computes feature attributions for the model output (value or
    action log-probability) along a straight-line path from a baseline state to
    the observed state.  The attributions are aggregated into a single scalar
    importance score that can be used to rank states in the critical buffer.

    Parameters
    ----------
    target_model : callable
        A PyTorch module/callable that accepts observations and returns a
        scalar output (e.g. value or log-prob of the taken action).  The input
        must be a ``torch.Tensor`` with ``requires_grad=True``.
    baseline : Optional[np.ndarray]
        Baseline observation used by IG.  If ``None``, a zero vector matching
        the observation shape is used.
    n_steps : int
        Number of interpolation steps between baseline and input (default 50).
    method : str
        Riemann-sum approximation method: ``"riemann_trapezoidal"`` or
        ``"riemann_middle"`` (default ``"riemann_trapezoidal"``).
    aggregator : str
        How to reduce the per-feature attributions to a scalar:
        ``"sum"`` (default), ``"mean"``, or ``"max_abs"``.
    """

    def __init__(
        self,
        target_model: Callable[[torch.Tensor], torch.Tensor],
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50,
        method: str = "riemann_trapezoidal",
        aggregator: str = "sum",
    ) -> None:
        self.target_model = target_model
        self.baseline = baseline
        self.n_steps = max(1, int(n_steps))
        self.method = method
        self.aggregator = aggregator

    def _get_baseline(self, obs: torch.Tensor) -> torch.Tensor:
        if self.baseline is None:
            return torch.zeros_like(obs)
        base = torch.as_tensor(self.baseline, dtype=obs.dtype, device=obs.device)
        if base.dim() == 1 and obs.dim() == 2:
            base = base.unsqueeze(0).expand_as(obs)
        return base

    def _interpolate(self, obs: torch.Tensor) -> torch.Tensor:
        """Build the straight-line path from baseline to obs."""
        baseline = self._get_baseline(obs)
        alphas = torch.linspace(0.0, 1.0, self.n_steps + 1, device=obs.device)
        # shape: (n_steps+1, 1, obs_dim) or (n_steps+1, batch, obs_dim)
        alphas = alphas.view(-1, *([1] * obs.dim()))
        path = baseline.unsqueeze(0) + alphas * (obs.unsqueeze(0) - baseline.unsqueeze(0))
        return path.view(-1, *obs.shape[1:])

    def _compute_gradients(self, path: torch.Tensor) -> torch.Tensor:
        """Compute gradients of target_model(path) w.r.t. path inputs."""
        path = path.detach().clone().requires_grad_(True)
        outputs = self.target_model(path)
        if outputs.dim() > 1:
            outputs = outputs.sum()
        outputs.backward()
        if path.grad is None:
            return torch.zeros_like(path)
        return path.grad.detach()

    def _riemann_sum(self, grads: torch.Tensor) -> torch.Tensor:
        """Approximate the integral using the selected Riemann method."""
        if self.method == "riemann_trapezoidal":
            # trapezoidal rule: average adjacent gradients and sum
            grads_mid = (grads[:-1] + grads[1:]) / 2.0
            return grads_mid.sum(dim=0)
        # riemann_middle / left
        return grads[:-1].sum(dim=0)

    def attribute(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute Integrated Gradients attributions for ``observation``.

        Parameters
        ----------
        observation : np.ndarray or torch.Tensor
            A single observation of shape ``(obs_dim,)`` or a batch of shape
            ``(batch_size, obs_dim)``.

        Returns
        -------
        np.ndarray
            Per-feature attributions with the same shape as ``observation``.
        """
        obs = torch.as_tensor(observation, dtype=torch.float32)
        was_single = obs.dim() == 1
        if was_single:
            obs = obs.unsqueeze(0)

        path = self._interpolate(obs)
        grads = self._compute_gradients(path)
        # Reshape gradients back to path shape
        grads = grads.view(self.n_steps + 1, obs.shape[0], obs.shape[1])
        integrated_grads = self._riemann_sum(grads)

        # Multiply by (input - baseline) to obtain IG attributions
        baseline = self._get_baseline(obs)
        integrated_grads = integrated_grads * (obs - baseline)

        if was_single:
            integrated_grads = integrated_grads.squeeze(0)
        return integrated_grads.detach().cpu().numpy()

    def predict(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Return a scalar criticality score for ``observation``.

        The score is computed by aggregating the IG attributions across the
        observation features.
        """
        attrs = self.attribute(observation)
        if attrs.ndim == 1:
            attrs = attrs[np.newaxis, :]

        if self.aggregator == "sum":
            scores = attrs.sum(axis=1)
        elif self.aggregator == "mean":
            scores = attrs.mean(axis=1)
        elif self.aggregator == "max_abs":
            scores = np.max(np.abs(attrs), axis=1)
        else:
            scores = attrs.sum(axis=1)

        # Normalize scores to [0, 1] per batch using min-max scaling
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s > 1e-8:
            scores = (scores - min_s) / (max_s - min_s)
        else:
            scores = np.ones_like(scores) * 0.5
        return scores

    def __call__(self, observation: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return self.predict(observation)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline,
            "n_steps": self.n_steps,
            "method": self.method,
            "aggregator": self.aggregator,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.baseline = state_dict.get("baseline", self.baseline)
        self.n_steps = state_dict.get("n_steps", self.n_steps)
        self.method = state_dict.get("method", self.method)
        self.aggregator = state_dict.get("aggregator", self.aggregator)


def _make_action_logit_fn(
    policy: Any,
    action: Optional[Union[int, np.ndarray, torch.Tensor]] = None,
    discrete: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a scalar-valued callable from a policy for IG attribution.

    The returned callable takes observations and returns the log-probability of
    the executed action (discrete) or the mean log-density (continuous).  If
    ``action`` is not provided, the policy's most likely action is used.
    """

    def fn(obs: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(True):
            if hasattr(policy, "evaluate_actions"):
                # TorchTargetPolicy interface
                if action is None:
                    act, _ = policy.predict(obs.detach().cpu().numpy(), deterministic=True)
                    act_t = torch.as_tensor(act, dtype=torch.long if discrete else torch.float32, device=obs.device)
                else:
                    act_t = torch.as_tensor(action, dtype=torch.long if discrete else torch.float32, device=obs.device)
                    if act_t.dim() == 0 and obs.dim() == 2:
                        act_t = act_t.unsqueeze(0).expand(obs.shape[0])
                _, log_prob, _ = policy.evaluate_actions(obs, act_t)
                return log_prob
            elif hasattr(policy, "forward"):
                # Direct actor-critic module
                logits, _ = policy.forward(obs)
                if discrete:
                    dist = torch.distributions.Categorical(logits=logits)
                    if action is None:
                        act_t = dist.probs.argmax(dim=-1)
                    else:
                        act_t = torch.as_tensor(action, dtype=torch.long, device=obs.device)
                        if act_t.dim() == 0 and obs.dim() == 2:
                            act_t = act_t.unsqueeze(0).expand(obs.shape[0])
                    return dist.log_prob(act_t)
                else:
                    mean, log_std = logits[:, : logits.shape[1] // 2], logits[:, logits.shape[1] // 2 :]
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    if action is None:
                        act_t = mean
                    else:
                        act_t = torch.as_tensor(action, dtype=torch.float32, device=obs.device)
                        if act_t.dim() == 1 and obs.dim() == 2:
                            act_t = act_t.unsqueeze(0).expand_as(mean)
                    return dist.log_prob(act_t).sum(dim=-1)
            else:
                raise ValueError("policy must provide evaluate_actions or forward")

    return fn


def integrated_gradients(
    observation: Union[np.ndarray, torch.Tensor],
    policy: Any,
    action: Optional[Union[int, np.ndarray, torch.Tensor]] = None,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 50,
    aggregator: str = "sum",
    discrete: bool = True,
) -> np.ndarray:
    """Functional IG scorer for a target policy.

    Parameters
    ----------
    observation : np.ndarray or torch.Tensor
        State(s) to score.
    policy : BaseTargetPolicy or nn.Module
        Target policy providing ``evaluate_actions`` or ``forward``.
    action : optional
        Action executed in ``observation``.  If ``None``, the deterministic
        policy action is used.
    baseline : optional np.ndarray
        IG baseline observation.
    n_steps : int
        Number of IG interpolation steps.
    aggregator : str
        Attribution aggregation method.
    discrete : bool
        Whether the action space is discrete.

    Returns
    -------
    np.ndarray
        Scalar criticality score(s) in ``[0, 1]``.
    """
    target_fn = _make_action_logit_fn(policy, action=action, discrete=discrete)
    ig = IntegratedGradients(
        target_model=target_fn,
        baseline=baseline,
        n_steps=n_steps,
        aggregator=aggregator,
    )
    return ig.predict(observation)
