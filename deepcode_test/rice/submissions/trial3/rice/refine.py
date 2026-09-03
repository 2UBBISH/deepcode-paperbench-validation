"""
RICE Refining Algorithm

Implements the main RICE (Refining via Critical State Explanation) algorithm:
1. Collect critical states using a trained mask network
2. Create a mixed initial state distribution
3. Refine the policy using PPO with RND exploration bonus

Theoretical guarantees (from paper):
- Theorem 3.3: η(π̄) ≤ η(π) under Assumption 3.1
- Lemma 3.5: MaskNet-based sampling is equivalent to sampling from a better policy π̂
- Theorem 3.6: After refining, V^π*(ρ) - V^π'(ρ) ≤ O(ε/(1-γ)² ||d_ρ^π* / d_ρ^π̂||_∞)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, Categorical
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
import copy
import os
import pickle
import warnings

from .mask_net import MaskNetwork
from .rnd import RNDModule, BonusNormalizer, compute_rnd_bonus_batch
from .utils import (
    TrajectoryBuffer, collect_trajectories, compute_gae, compute_returns,
    evaluate_policy, set_seed, to_tensor, to_numpy, orthogonal_init,
    save_state_dict, load_state_dict,
)
from .env_wrappers import (
    StateSaveWrapper, make_state_saveable, save_env_state,
    restore_env_state, reset_env_to_state,
)


class RICERefine:
    """Main RICE refining algorithm orchestrator.

    Three-phase pipeline:
    Phase 1: Collect critical states using trained mask network
    Phase 2: Mixed initial distribution (p: critical state, 1-p: default reset)
    Phase 3: Refining loop with PPO + RND exploration bonus

    Args:
        env: Gym environment (will be wrapped for state save/restore)
        target_policy: Pre-trained policy nn.Module
        mask_network: Trained MaskNetwork for importance scoring
        state_dim: State space dimension
        action_dim: Action space dimension
        discrete_action: Whether action space is discrete
        num_discrete_actions: Number of discrete actions (if discrete)
        device: Computation device
        p_mixed: Probability of resetting to critical state (default 0.25)
        lambda_rnd: RND exploration bonus weight (default 0.01)
        rnd_hidden_sizes: RND network hidden sizes (default (64,64))
        rnd_embedding_dim: RND embedding dimension (default 64)
        rnd_lr: RND predictor learning rate (default 1e-4)
        ppo_lr: PPO learning rate (default 3e-4)
        ppo_epochs: PPO epochs per update (default 10)
        ppo_batch_size: PPO mini-batch size (default 64)
        ppo_clip_epsilon: PPO clip epsilon (default 0.2)
        gamma: Discount factor (default 0.99)
        gae_lambda: GAE lambda (default 0.95)
        value_loss_coef: Value loss coefficient (default 0.5)
        entropy_coef: Entropy coefficient (default 0.01)
        max_grad_norm: Max gradient norm (default 0.5)
        normalize_advantages: Normalize advantages (default True)
        normalize_obs: Normalize obs for RND (default True)
        normalize_bonus: Normalize RND bonus (default True)
        top_k_per_episode: Critical states per episode (default 1)
        num_critical_episodes: Episodes for collection (default 100)
        policy_std: Fixed std for continuous actions (default 0.5)
        value_hidden_sizes: Value network hidden sizes (default (128,128))
    """

    def __init__(
        self,
        env,
        target_policy: nn.Module,
        mask_network: MaskNetwork,
        state_dim: int,
        action_dim: int,
        discrete_action: bool = False,
        num_discrete_actions: Optional[int] = None,
        device: str = "cpu",
        p_mixed: float = 0.25,
        lambda_rnd: float = 0.01,
        rnd_hidden_sizes: Tuple[int, ...] = (64, 64),
        rnd_embedding_dim: int = 64,
        rnd_lr: float = 1e-4,
        ppo_lr: float = 3e-4,
        ppo_epochs: int = 10,
        ppo_batch_size: int = 64,
        ppo_clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        normalize_obs: bool = True,
        normalize_bonus: bool = True,
        top_k_per_episode: int = 1,
        num_critical_episodes: int = 100,
        policy_std: float = 0.5,
        value_hidden_sizes: Tuple[int, ...] = (128, 128),
    ):
        self.env = make_state_saveable(env)
        self._base_env = env

        self.target_policy = target_policy
        self.mask_network = mask_network
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_action = discrete_action
        self.num_discrete_actions = num_discrete_actions
        self.device = device
        self.policy_std = policy_std

        self.p_mixed = p_mixed
        self.lambda_rnd = lambda_rnd
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_lr = ppo_lr
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.top_k_per_episode = top_k_per_episode
        self.num_critical_episodes = num_critical_episodes

        # RND module
        self.rnd_module = RNDModule(
            state_dim=state_dim,
            hidden_sizes=rnd_hidden_sizes,
            embedding_dim=rnd_embedding_dim,
            learning_rate=rnd_lr,
            device=device,
            normalize_obs=normalize_obs,
        )
        self.bonus_normalizer = BonusNormalizer() if normalize_bonus else None

        self.critical_states: List[Dict[str, Any]] = []

        # Move to device
        self.target_policy.to(device)
        self.mask_network.to(device)

        # Copy target policy for refining
        self.current_policy = copy.deepcopy(target_policy)
        self.current_policy.to(device)
        self.current_policy.train()

        # Value network
        self.value_network = self._build_value_network(value_hidden_sizes)
        self.value_network.to(device)
        self.value_network.train()

        # Optimizer
        self.ppo_optimizer = Adam(
            list(self.current_policy.parameters())
            + list(self.value_network.parameters()),
            lr=ppo_lr,
        )

        self.total_steps_done = 0
        self.iteration = 0

    def _build_value_network(self, hidden_sizes: Tuple[int, ...]) -> nn.Module:
        layers = []
        input_dim = self.state_dim
        for hidden_size in hidden_sizes:
            linear = nn.Linear(input_dim, hidden_size)
            orthogonal_init(linear, gain=np.sqrt(2))
            layers.append(linear)
            layers.append(nn.Tanh())
            input_dim = hidden_size
        output_layer = nn.Linear(input_dim, 1)
        orthogonal_init(output_layer, gain=1.0)
        layers.append(output_layer)
        return nn.Sequential(*layers)

    # ==================================================================
    # Phase 1: Critical State Collection
    # ==================================================================

    def collect_critical_states(
        self,
        num_episodes: Optional[int] = None,
        max_steps_per_episode: int = 1000,
        verbose: bool = True,
        deterministic_target: bool = False,
    ) -> List[Dict[str, Any]]:
        """Collect critical states by running target policy and scoring with mask network.

        For each episode, saves env state at each step, computes importance score
        via mask network, and selects top-k states per episode.
        """
        if num_episodes is None:
            num_episodes = self.num_critical_episodes

        all_critical_states = []
        all_importance_scores = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            truncated = False
            step = 0
            episode_states: List[Dict[str, Any]] = []

            while not done and not truncated and step < max_steps_per_episode:
                env_state = save_env_state(self.env)
                obs_tensor = to_tensor(obs, self.device).unsqueeze(0)
                with torch.no_grad():
                    importance_score = self.mask_network.get_importance_score(obs_tensor).item()

                episode_states.append({
                    "env_state": env_state,
                    "importance_score": importance_score,
                    "observation": obs.copy(),
                    "episode": episode,
                    "step": step,
                })
                all_importance_scores.append(importance_score)

                action = self._get_target_action(obs, deterministic_target)
                obs, reward, done, truncated, info = self._env_step(action)
                step += 1

            episode_states.sort(key=lambda x: x["importance_score"], reverse=True)
            all_critical_states.extend(episode_states[: self.top_k_per_episode])

            if verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
                print(f"[Critical States] Ep {episode+1}/{num_episodes} | "
                      f"Collected: {len(all_critical_states)} | Steps: {step}")

        self.critical_states = all_critical_states

        if verbose:
            print(f"\n{'='*60}")
            print(f"Critical State Collection Complete: {len(self.critical_states)} states")
            if all_importance_scores:
                scores = np.array(all_importance_scores)
                print(f"All scores: mean={scores.mean():.4f} std={scores.std():.4f} "
                      f"min={scores.min():.4f} max={scores.max():.4f}")
            if self.critical_states:
                crit = [s["importance_score"] for s in self.critical_states]
                print(f"Critical scores: mean={np.mean(crit):.4f} std={np.std(crit):.4f} "
                      f"min={np.min(crit):.4f} max={np.max(crit):.4f}")
            print(f"{'='*60}\n")

        return self.critical_states

    def _get_target_action(self, obs: np.ndarray, deterministic: bool = False) -> Any:
        obs_tensor = to_tensor(obs, self.device).unsqueeze(0)
        with torch.no_grad():
            if self.discrete_action:
                logits = self.target_policy(obs_tensor)
                dist = Categorical(logits=logits)
                if deterministic:
                    return dist.probs.argmax(dim=-1).item()
                return dist.sample().item()
            else:
                out = self.target_policy(obs_tensor)
                if isinstance(out, tuple):
                    mean, std = out
                else:
                    mean, std = out, torch.ones_like(out) * self.policy_std
                dist = Normal(mean, std)
                if deterministic:
                    return mean.squeeze(0).cpu().numpy()
                return dist.sample().squeeze(0).cpu().numpy()

    def _env_step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated or truncated, truncated, info
        obs, reward, done, info = result
        return obs, reward, done, False, info

    # ==================================================================
    # Phase 2: Mixed Initial Distribution
    # ==================================================================

    def _reset_with_mixed_distribution(self) -> np.ndarray:
        """Reset env: with prob p_mixed restore to critical state, else default reset."""
        if len(self.critical_states) > 0 and np.random.random() < self.p_mixed:
            idx = np.random.randint(0, len(self.critical_states))
            return reset_env_to_state(self.env, self.critical_states[idx]["env_state"])
        obs, _ = self.env.reset()
        return obs

    # ==================================================================
    # Policy Interface
    # ==================================================================

    def _get_policy_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[Any, float, float]:
        """Get (action, log_prob, value) from current policy."""
        obs_tensor = to_tensor(obs, self.device).unsqueeze(0)
        with torch.no_grad():
            if self.discrete_action:
                logits = self.current_policy(obs_tensor)
                dist = Categorical(logits=logits)
                if deterministic:
                    action = dist.probs.argmax(dim=-1).item()
                else:
                    action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            else:
                out = self.current_policy(obs_tensor)
                if isinstance(out, tuple):
                    mean, std = out
                else:
                    mean, std = out, torch.ones_like(out) * self.policy_std
                dist = Normal(mean, std)
                if deterministic:
                    action = mean.squeeze(0).cpu().numpy()
                else:
                    action = dist.sample().squeeze(0).cpu().numpy()
                log_prob = dist.log_prob(to_tensor(action, self.device)).sum(dim=-1).item()
            value = self.value_network(obs_tensor).item()
        return action, log_prob, value

    def _evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate (log_probs, values, entropy) for a batch."""
        if self.discrete_action:
            logits = self.current_policy(obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        else:
            out = self.current_policy(obs)
            if isinstance(out, tuple):
                mean, std = out
            else:
                mean, std = out, torch.ones_like(out) * self.policy_std
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        values = self.value_network(obs).squeeze(-1)
        return log_probs, values, entropy

    # ==================================================================
    # PPO Update
    # ==================================================================

    def _ppo_update(self, buffer: TrajectoryBuffer) -> Dict[str, float]:
        """PPO clipped objective update on collected trajectories."""
        data = buffer.get_all()
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        old_values = data["values"]
        old_log_probs = data["log_probs"]

        advantages, returns = compute_gae(
            rewards=rewards, values=old_values, dones=dones,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        if self.normalize_advantages:
            adv_std = advantages.std()
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        states_t = to_tensor(states, self.device)
        actions_t = to_tensor(actions, self.device)
        if self.discrete_action:
            actions_t = actions_t.long()
        advantages_t = to_tensor(advantages, self.device)
        returns_t = to_tensor(returns, self.device)
        old_log_probs_t = to_tensor(old_log_probs, self.device)

        total_samples = len(states)
        indices = np.arange(total_samples)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "total_loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        num_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, self.ppo_batch_size):
                end = start + self.ppo_batch_size
                bi = indices[start:end]

                b_states = states_t[bi]
                b_actions = actions_t[bi]
                b_adv = advantages_t[bi]
                b_returns = returns_t[bi]
                b_old_lp = old_log_probs_t[bi]

                new_lp, values, entropy = self._evaluate_actions(b_states, b_actions)

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon,
                                   1.0 + self.ppo_clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, b_returns)
                total_loss = (policy_loss + self.value_loss_coef * value_loss
                              - self.entropy_coef * entropy)

                self.ppo_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.current_policy.parameters())
                    + list(self.value_network.parameters()),
                    self.max_grad_norm,
                )
                self.ppo_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.ppo_clip_epsilon).float().mean().item()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["total_loss"] += total_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_fraction"] += clip_frac
                num_updates += 1

        if num_updates > 0:
            for k in stats:
                stats[k] /= num_updates
        return stats

    # ==================================================================
    # Phase 3: Refining Loop
    # ==================================================================

    def refine(
        self,
        total_steps: int = 1_000_000,
        steps_per_iteration: int = 2048,
        eval_interval: int = 10,
        eval_episodes: int = 10,
        rnd_update_epochs: int = 4,
        verbose: bool = True,
        save_path: Optional[str] = None,
        save_interval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the RICE refining loop.

        Iteratively: reset with mixed distribution, collect trajectory with RND bonus,
        update RND predictor, update policy via PPO.
        """
        if len(self.critical_states) == 0:
            raise ValueError("No critical states. Call collect_critical_states() first.")

        if save_interval is None:
            save_interval = eval_interval

        history: Dict[str, List[Any]] = {
            "iteration": [], "mean_reward": [], "std_reward": [],
            "policy_loss": [], "value_loss": [], "entropy": [],
            "rnd_loss": [], "total_steps": [],
        }

        total_steps_done = 0
        iteration = 0
        best_eval_reward = -float("inf")

        if verbose:
            print(f"\n{'='*60}\nStarting RICE Refining\n{'='*60}")
            print(f"Total steps: {total_steps} | Steps/iter: {steps_per_iteration}")
            print(f"p_mixed: {self.p_mixed} | lambda_rnd: {self.lambda_rnd}")
            print(f"Critical states: {len(self.critical_states)}\n{'='*60}\n")

        while total_steps_done < total_steps:
            iteration += 1

            # Collect trajectory
            buffer = TrajectoryBuffer(
                state_dim=self.state_dim, action_dim=self.action_dim,
                capacity=steps_per_iteration, discrete_action=self.discrete_action,
                device=self.device,
            )

            obs = self._reset_with_mixed_distribution()
            done = False
            truncated = False
            iteration_states: List[np.ndarray] = []

            for step in range(steps_per_iteration):
                action, log_prob, value = self._get_policy_action(obs)
                next_obs, env_reward, done, truncated, info = self._env_step(action)

                # RND bonus
                obs_t = to_tensor(obs, self.device).unsqueeze(0)
                with torch.no_grad():
                    rnd_raw = self.rnd_module.compute_bonus(obs_t).item()
                if self.bonus_normalizer is not None:
                    self.bonus_normalizer.update(np.array([rnd_raw]))
                    rnd_bonus = self.bonus_normalizer.normalize(np.array([rnd_raw]))[0]
                else:
                    rnd_bonus = rnd_raw

                total_reward = env_reward + self.lambda_rnd * rnd_bonus

                buffer.add(
                    state=obs, action=action, reward=total_reward, done=done,
                    value=value, log_prob=log_prob, mask=0.0, next_state=next_obs,
                    info={"env_reward": env_reward, "rnd_bonus_raw": rnd_raw,
                          "rnd_bonus": rnd_bonus},
                )
                iteration_states.append(obs.copy())
                obs = next_obs

                if done or truncated:
                    obs = self._reset_with_mixed_distribution()
                    done = False
                    truncated = False

            total_steps_done += steps_per_iteration

            # Update RND
            if iteration_states:
                rnd_stats = self.rnd_module.update_on_trajectory(
                    np.array(iteration_states), num_epochs=rnd_update_epochs)
            else:
                rnd_stats = {"loss": 0.0}

            # PPO update
            ppo_stats = self._ppo_update(buffer)

            # Log
            history["iteration"].append(iteration)
            history["policy_loss"].append(ppo_stats["policy_loss"])
            history["value_loss"].append(ppo_stats["value_loss"])
            history["entropy"].append(ppo_stats["entropy"])
            history["rnd_loss"].append(rnd_stats["loss"])
            history["total_steps"].append(total_steps_done)

            # Evaluate
            if iteration % eval_interval == 0 or total_steps_done >= total_steps:
                eval_stats = evaluate_policy(
                    env=self.env,
                    policy_fn=lambda o: self._get_policy_action(o, deterministic=True)[0],
                    num_episodes=eval_episodes, deterministic=True,
                )
                history["mean_reward"].append(eval_stats["mean_reward"])
                history["std_reward"].append(eval_stats["std_reward"])

                if eval_stats["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_stats["mean_reward"]
                    if save_path:
                        self.save(os.path.join(save_path, "best.pt"))

                if verbose:
                    print(f"Iter {iteration:4d} | Steps: {total_steps_done:8d} | "
                          f"Eval: {eval_stats['mean_reward']:8.2f} ± "
                          f"{eval_stats['std_reward']:6.2f} | "
                          f"P Loss: {ppo_stats['policy_loss']:.4f} | "
                          f"V Loss: {ppo_stats['value_loss']:.4f} | "
                          f"RND Loss: {rnd_stats['loss']:.6f}")
            else:
                # Pad history for consistent lengths
                if len(history["mean_reward"]) < len(history["iteration"]):
                    last_r = history["mean_reward"][-1] if history["mean_reward"] else 0.0
                    last_s = history["std_reward"][-1] if history["std_reward"] else 0.0
                    history["mean_reward"].append(last_r)
                    history["std_reward"].append(last_s)

            # Save checkpoint
            if save_path and iteration % save_interval == 0:
                self.save(os.path.join(save_path, f"checkpoint_{iteration}.pt"))

        if save_path:
            self.save(os.path.join(save_path, "final.pt"))

        if verbose:
            print(f"\n{'='*60}\nRefining Complete! Best eval: {best_eval_reward:.2f}\n{'='*60}")

        return history

    # ==================================================================
    # Save / Load
    # ==================================================================

    def save(self, path: str) -> None:
        """Save refined policy and RICE state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.current_policy.state_dict(),
            "value_state_dict": self.value_network.state_dict(),
            "rnd_state_dict": self.rnd_module.state_dict(),
            "critical_states": self.critical_states,
            "config": {
                "p_mixed": self.p_mixed, "lambda_rnd": self.lambda_rnd,
                "state_dim": self.state_dim, "action_dim": self.action_dim,
                "discrete_action": self.discrete_action,
                "num_discrete_actions": self.num_discrete_actions,
                "policy_std": self.policy_std,
            },
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load a refined policy and RICE state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_network.load_state_dict(checkpoint["value_state_dict"])
        self.rnd_module.load_state_dict(checkpoint["rnd_state_dict"])
        self.critical_states = checkpoint["critical_states"]
        if "config" in checkpoint:
            for k, v in checkpoint["config"].items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def get_policy(self) -> nn.Module:
        """Return the refined policy."""
        return self.current_policy

    def save_critical_states(self, path: str) -> None:
        """Save critical states to disk (pickle)."""
        save_state_dict({"critical_states": self.critical_states}, path)

    def load_critical_states(self, path: str) -> None:
        """Load critical states from disk (pickle)."""
        data = load_state_dict(path)
        self.critical_states = data["critical_states"]


def refine_policy(
    env,
    target_policy: nn.Module,
    mask_network: MaskNetwork,
    state_dim: int,
    action_dim: int,
    discrete_action: bool = False,
    num_discrete_actions: Optional[int] = None,
    device: str = "cpu",
    p_mixed: float = 0.25,
    lambda_rnd: float = 0.01,
    total_steps: int = 1_000_000,
    steps_per_iteration: int = 2048,
    num_critical_episodes: int = 100,
    top_k_per_episode: int = 1,
    eval_interval: int = 10,
    eval_episodes: int = 10,
    verbose: bool = True,
    save_path: Optional[str] = None,
    **kwargs,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Convenience function for end-to-end RICE refining.

    Returns (refined_policy, training_history).
    """
    refiner = RICERefine(
        env=env, target_policy=target_policy, mask_network=mask_network,
        state_dim=state_dim, action_dim=action_dim,
        discrete_action=discrete_action, num_discrete_actions=num_discrete_actions,
        device=device, p_mixed=p_mixed, lambda_rnd=lambda_rnd, **kwargs,
    )

    if verbose:
        print("=" * 60)
        print("Phase 1: Collecting Critical States")
        print("=" * 60)

    refiner.collect_critical_states(
        num_episodes=num_critical_episodes, verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Phase 2: Refining Policy")
        print("=" * 60)

    history = refiner.refine(
        total_steps=total_steps, steps_per_iteration=steps_per_iteration,
        eval_interval=eval_interval, eval_episodes=eval_episodes,
        verbose=verbose, save_path=save_path,
    )

    return refiner.get_policy(), history