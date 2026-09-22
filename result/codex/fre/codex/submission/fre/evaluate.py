"""Zero-shot evaluation of a pre-trained FRE agent (Section 5.2, Table 1).

Evaluation protocol (Section 5.2 and the addendum):

  * each downstream task is specified by 32 ``(state, reward)`` samples.  The
    states are drawn uniformly from the offline dataset and the reward is the
    task's ground-truth reward at that state;
  * those samples are encoded into a latent ``z`` with the frozen FRE encoder;
  * the FRE-conditioned policy is rolled out for at most ``max_episode_steps``
    (2000 on AntMaze, 1000 on ExORL) with no further training;
  * each task is evaluated over 20 episodes and each agent is trained with 5
    seeds, reporting the mean across episodes and the standard deviation
    across seeds;
  * returns are normalised to the [0, 100] range.

This module is written so that it can drive either the real D4RL / DMC
environments or a lightweight stub environment (used in the unit tests).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fre.configs import EvalConfig, TrainConfig
from fre.datasets import OfflineDataset
from fre.fre import FRE, FREConfig
from fre.iql import IQLAgent, IQLConfig
from fre.reward_functions import discretize_reward
from fre.tasks.base import EvalTask, TaskSuite
from fre.training import make_preprocess_fn, resolve_device


class FREPolicy:
    """Latent-conditioned policy wrapper around the trained FRE + IQL agent."""

    def __init__(
        self,
        model: FRE,
        agent: IQLAgent,
        preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]],
        device: torch.device,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.agent = agent
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.state_mean = None if state_mean is None else np.asarray(state_mean, dtype=np.float32)
        self.state_std = None if state_std is None else np.asarray(state_std, dtype=np.float32)

    def _prepare(self, states: np.ndarray) -> np.ndarray:
        """Apply the same preprocessing the trained model saw during training."""
        states = np.asarray(states, dtype=np.float32)
        if self.preprocess_fn is not None:
            states = self.preprocess_fn(states)
        if self.state_mean is not None and self.state_std is not None:
            states = (states - self.state_mean) / self.state_std
        return states

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        dataset: OfflineDataset,
        device: str = "cpu",
    ) -> "FREPolicy":
        torch_device = resolve_device(device)
        payload = torch.load(checkpoint_path, map_location=torch_device)
        config = TrainConfig(**payload["config"])
        model = FRE(
            FREConfig(
                state_dim=dataset.encoder_obs_dim,
                z_dim=config.z_dim,
                state_embed_dim=config.state_embed_dim,
                reward_embed_dim=config.reward_embed_dim,
                d_model=config.d_model,
                n_layers=config.encoder_layers,
                n_heads=config.encoder_heads,
                encoder_mlp_dim=config.encoder_mlp_dim,
                num_reward_bins=config.num_reward_bins,
                decoder_hidden_dims=config.decode_hidden_dims
                if hasattr(config, "decode_hidden_dims")
                else config.decoder_hidden_dims,
                beta=config.beta,
            )
        ).to(torch_device)
        model.encoder.load_state_dict(payload["encoder"])
        model.decoder.load_state_dict(payload["decoder"])
        model.eval()

        agent = IQLAgent(
            IQLConfig(
                obs_dim=dataset.obs_dim,
                action_dim=dataset.action_dim,
                z_dim=config.z_dim,
                hidden_dims=config.rl_hidden_dims,
                discount=config.discount,
                expectile=config.expectile,
                awr_temperature=config.awr_temperature,
                target_update_rate=config.target_update_rate,
            ),
            device=torch_device,
        )
        agent.load_state_dict(payload["agent"])
        return cls(
            model,
            agent,
            make_preprocess_fn(config, dataset.obs_dim),
            torch_device,
            state_mean=payload.get("state_mean"),
            state_std=payload.get("state_std"),
        )

    # -- task conditioning ---------------------------------------------------------
    def encode_task(
        self,
        task: EvalTask,
        dataset: OfflineDataset,
        num_samples: int = 32,
        rng: Optional[np.random.Generator] = None,
        states: Optional[np.ndarray] = None,
        sample: bool = False,
        num_z_samples: int = 1,
    ) -> np.ndarray:
        """Encode ``num_samples`` ``(state, task_reward)`` pairs into a latent ``z``.

        ``sample=False`` (the default) uses the posterior mean, the standard
        choice for a VAE encoder.  Passing ``sample=True`` draws
        ``num_z_samples`` latents from the posterior and averages them, which is
        the more faithful reading of "encode samples of their reward functions
        into the latent space" and is more robust when the posterior is wide.
        """
        rng = rng or np.random.default_rng(0)
        if states is None:
            states = dataset.encoder_observations[rng.integers(0, len(dataset), size=num_samples)]
        states = np.asarray(states, dtype=np.float32)
        rewards = np.asarray(task.reward(states), dtype=np.float32)
        states = self._prepare(states)
        with torch.no_grad():
            state_t = torch.as_tensor(states, device=self.device)
            bins = discretize_reward(
                torch.as_tensor(rewards, device=self.device),
                self.model.config.num_reward_bins,
                r_min=task.reward_range[0],
                r_max=task.reward_range[1],
            )
            if not sample:
                z = self.model.encode(state_t.unsqueeze(0), bins.unsqueeze(0), sample=False)
            else:
                z = torch.stack(
                    [
                        self.model.encode(state_t.unsqueeze(0), bins.unsqueeze(0), sample=True)
                        for _ in range(max(1, num_z_samples))
                    ]
                ).mean(dim=0)
        return z.squeeze(0).cpu().numpy()

    # -- acting --------------------------------------------------------------------
    def act(self, obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs = self._prepare(obs)
        obs_t = torch.as_tensor(obs, device=self.device)
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32).reshape(1, -1), device=self.device)
        with torch.no_grad():
            return self.agent.act(obs_t, z_t).cpu().numpy()


# --------------------------------------------------------------------------------------
# Rollout driver
# --------------------------------------------------------------------------------------
def rollout_episode(
    env,
    task: EvalTask,
    policy: FREPolicy,
    z: np.ndarray,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    """Run one episode and return the cumulative *task* reward."""
    max_steps = max_episode_steps or task.max_episode_steps
    obs = np.asarray(task.reset_env(env, seed), dtype=np.float32).reshape(-1)
    total = 0.0
    for _ in range(max_steps):
        action = policy.act(obs, z).reshape(-1)
        step_out = env.step(action)
        obs = np.asarray(step_out[0], dtype=np.float32).reshape(-1)
        total += float(np.asarray(task.reward(obs.reshape(1, -1))).reshape(-1)[0])
        if len(step_out) > 2 and bool(np.asarray(step_out[2]).reshape(-1)[0]):
            break
    return total


def evaluate_task(
    env,
    task: EvalTask,
    policy: FREPolicy,
    dataset: OfflineDataset,
    eval_config: EvalConfig,
    seed: int = 0,
) -> Dict[str, float]:
    """Evaluate one downstream task: encode, roll out, normalise."""
    rng = np.random.default_rng(seed)
    z = policy.encode_task(task, dataset, num_samples=eval_config.num_encode_pairs, rng=rng)
    returns = [
        rollout_episode(env, task, policy, z, seed=seed * 1000 + ep)
        for ep in range(eval_config.num_episodes)
    ]
    raw = np.asarray(returns, dtype=np.float64)
    normalised = task.normalize(raw)
    return {
        "task": task.name,
        "raw_return_mean": float(raw.mean()),
        "raw_return_std": float(raw.std()),
        "normalized_mean": float(normalised.mean()),
        "normalized_std": float(normalised.std()),
    }


def evaluate_suite(
    env_factory: Callable[[EvalTask], object],
    suite: TaskSuite,
    policy: FREPolicy,
    dataset: OfflineDataset,
    eval_config: EvalConfig,
    seed: int = 0,
) -> Dict[str, float]:
    """Evaluate every task in a suite and average (one column of Table 1)."""
    per_task = []
    for i, task in enumerate(suite.tasks):
        env = env_factory(task)
        per_task.append(evaluate_task(env, task, policy, dataset, eval_config, seed=seed + i))
    scores = np.asarray([t["normalized_mean"] for t in per_task], dtype=np.float64)
    return {
        "suite": suite.name,
        "mean": float(scores.mean()),
        "std_across_tasks": float(scores.std()),
        "per_task": per_task,
    }


def aggregate_seeds(suite_results: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average a suite across training seeds and report the across-seed std.

    Table 1 reports the standard deviation over 5 seeds of the per-seed score.
    """
    means = np.asarray([r["mean"] for r in suite_results], dtype=np.float64)
    return {
        "suite": suite_results[0]["suite"],
        "mean": float(means.mean()),
        "std": float(means.std()),
        "num_seeds": len(means),
    }


def save_results(results: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
