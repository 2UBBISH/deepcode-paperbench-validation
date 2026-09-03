"""Experiment IV: Refine a pre-trained SAC agent.

The paper trains a SAC agent on Hopper and then learns a PPO policy that
approximates the SAC policy via Generative Adversarial Imitation Learning
(GAIL). The resulting PPO policy is then refined with RICE and the baselines.
"""
import argparse
import json
import os
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rice.baselines import jsrl_finetune, ppo_finetune, statemask_r_finetune
from rice.env_utils import make_env
from rice.explanations import MaskExplanation, RandomExplanation, StateMaskExplanation
from rice.mask_network import MaskNetworkTrainer
from rice.refining import refine_rice
from rice.utils import set_seed


def collect_sac_demonstrations(
    sac_model: SAC,
    env: gym.Env,
    n_episodes: int = 50,
    max_steps: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect state-action demonstrations from a trained SAC agent."""
    observations = []
    actions = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action, _ = sac_model.predict(obs, deterministic=True)
            observations.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.float32)


class GAILDiscriminator(nn.Module):
    """Binary classifier for GAIL: distinguishes expert from policy samples."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_size = obs_dim + act_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h), nn.ReLU()])
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return torch.sigmoid(self.network(x))

    def reward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # Reward is log(D(s,a)) - log(1 - D(s,a)) as in Ho & Ermon (2016).
        d = self.forward(obs, act)
        return torch.log(d + 1e-8) - torch.log(1 - d + 1e-8)


class GAILRewardCallback(BaseCallback):
    """SB3 callback that replaces environment rewards with GAIL rewards.

    This is applied during PPO .learn() so that SB3 handles rollout collection
    and policy updates, while we update the discriminator and override rewards.
    """

    def __init__(
        self,
        discriminator: GAILDiscriminator,
        expert_obs: torch.Tensor,
        expert_actions: torch.Tensor,
        disc_optimizer: optim.Optimizer,
        device: torch.device,
        batch_size: int = 256,
        n_disc_epochs: int = 2,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.discriminator = discriminator
        self.expert_obs = expert_obs
        self.expert_actions = expert_actions
        self.disc_optimizer = disc_optimizer
        self.device = device
        self.batch_size = batch_size
        self.n_disc_epochs = n_disc_epochs

    def _on_rollout_end(self) -> None:
        """Called after SB3 has collected a rollout."""
        rollout = self.model.rollout_buffer
        obs = torch.as_tensor(rollout.observations, device=self.device)
        actions = torch.as_tensor(rollout.actions, device=self.device)
        with torch.no_grad():
            gail_rewards = self.discriminator.reward(obs, actions).cpu().numpy()
        rollout.rewards[:] = gail_rewards

        # Update discriminator.
        n_expert = self.expert_obs.shape[0]
        n_policy = obs.shape[0]
        bs = min(self.batch_size, min(n_expert, n_policy))
        for _ in range(self.n_disc_epochs):
            expert_idx = np.random.choice(n_expert, size=bs, replace=False)
            policy_idx = np.random.choice(n_policy, size=bs, replace=False)
            expert_pred = self.discriminator(
                self.expert_obs[expert_idx], self.expert_actions[expert_idx]
            )
            policy_pred = self.discriminator(obs[policy_idx], actions[policy_idx])
            loss = -(
                torch.log(expert_pred + 1e-8).mean()
                + torch.log(1 - policy_pred + 1e-8).mean()
            )
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

    def _on_step(self) -> bool:
        return True


def gail_imitation(
    env: gym.Env,
    expert_obs: np.ndarray,
    expert_actions: np.ndarray,
    total_timesteps: int = 500_000,
    lr: float = 3e-4,
    seed: int = 0,
) -> PPO:
    """Train a PPO policy to imitate a SAC expert via GAIL.

    We use Stable-Baselines3's PPO learner with a custom callback that replaces
    the environment reward with the GAIL discriminator reward and updates the
    discriminator on each rollout.
    """
    set_seed(seed)
    vec_env = DummyVecEnv([lambda: env])
    policy = PPO("MlpPolicy", vec_env, verbose=1, seed=seed)

    obs_dim = expert_obs.shape[-1]
    act_dim = expert_actions.shape[-1]
    discriminator = GAILDiscriminator(obs_dim, act_dim).to(policy.device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    expert_obs_t = torch.as_tensor(expert_obs, device=policy.device)
    expert_actions_t = torch.as_tensor(expert_actions, device=policy.device)

    callback = GAILRewardCallback(
        discriminator=discriminator,
        expert_obs=expert_obs_t,
        expert_actions=expert_actions_t,
        disc_optimizer=disc_optimizer,
        device=policy.device,
    )
    policy.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=True)
    return policy


def run_experiment_iv(
    env_id: str = "Hopper-v3",
    sac_timesteps: int = 1_000_000,
    gail_timesteps: int = 500_000,
    refine_timesteps: int = 500_000,
    output_dir: str = "results/sac_hopper",
    seed: int = 0,
) -> Dict[str, Any]:
    """Run Experiment IV: refine a SAC agent via GAIL-imitated PPO."""
    set_seed(seed)
    env = make_env(env_id, seed=seed)

    # Train SAC agent.
    sac_model = SAC("MlpPolicy", env, verbose=1, seed=seed)
    sac_model.learn(total_timesteps=sac_timesteps)

    # Collect demonstrations.
    expert_obs, expert_actions = collect_sac_demonstrations(sac_model, env)

    # Train PPO policy via GAIL.
    imitated_policy = gail_imitation(env, expert_obs, expert_actions, total_timesteps=gail_timesteps, seed=seed)

    # Train mask network on the imitated PPO policy.
    obs_dim = int(np.prod(env.observation_space.shape))
    mask_trainer = MaskNetworkTrainer(
        env=env,
        target_policy=imitated_policy,
        obs_dim=obs_dim,
        alpha=0.0001,
    )
    mask_trainer.train(total_timesteps=300_000, steps_per_iter=2048)

    hparams = {"p": 0.25, "lambda": 0.01, "alpha": 0.0001}

    results: Dict[str, Any] = {}

    # Baselines.
    results["PPO_finetune"] = evaluate_policy(
        env, ppo_finetune(env, imitated_policy, total_timesteps=refine_timesteps)
    )
    results["StateMask-R"] = evaluate_policy(
        env,
        statemask_r_finetune(
            env,
            imitated_policy,
            explanation=MaskExplanation(mask_trainer.mask_net),
            total_timesteps=refine_timesteps,
        ),
    )
    results["JSRL"] = evaluate_policy(
        env, jsrl_finetune(env, imitated_policy, total_timesteps=refine_timesteps)
    )
    # Fine-tune the original SAC agent with SAC algorithm.
    sac_finetuned = SAC("MlpPolicy", env, verbose=1, seed=seed)
    sac_finetuned.set_parameters(sac_model.get_parameters(), exact_match=True)
    sac_finetuned.learn(total_timesteps=refine_timesteps, reset_num_timesteps=False)
    results["SAC_finetune"] = evaluate_policy(env, sac_finetuned)

    # RICE.
    results["Ours"] = evaluate_policy(
        env,
        refine_rice(
            env,
            imitated_policy,
            mask_trainer.mask_net,
            total_timesteps=refine_timesteps,
            p=hparams["p"],
            lambda_rnd=hparams["lambda"],
            alpha=hparams["alpha"],
        ),
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_iv.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def evaluate_policy(
    env: gym.Env,
    policy: Any,
    n_eval_episodes: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    """Evaluate a policy by running several episodes."""
    rewards = []
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < 1000:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        rewards.append(total_reward)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "rewards": [float(r) for r in rewards],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Hopper-v3")
    parser.add_argument("--sac-timesteps", type=int, default=1_000_000)
    parser.add_argument("--gail-timesteps", type=int, default=500_000)
    parser.add_argument("--refine-timesteps", type=int, default=500_000)
    parser.add_argument("--output-dir", type=str, default="results/sac_hopper")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_experiment_iv(
        env_id=args.env_id,
        sac_timesteps=args.sac_timesteps,
        gail_timesteps=args.gail_timesteps,
        refine_timesteps=args.refine_timesteps,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
