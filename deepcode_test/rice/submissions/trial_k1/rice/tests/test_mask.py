"""Unit tests for the RICE mask network and perturbed-policy explanation module."""
from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from rice.agents.target_agent import TargetAgent, TargetAgentConfig, train_target_agent_sb3
from rice.agents.mask_network import (
    MaskNetwork,
    PerturbedPolicy,
    MaskTrainingConfig,
    make_mask_network,
    train_mask_network,
    collect_masked_rollouts,
    extract_critical_states,
)


@pytest.fixture(scope="module")
def target_agent() -> TargetAgent:
    """Train a small target policy on CartPole once for all mask tests."""
    env = gym.make("CartPole-v1")
    config = TargetAgentConfig(
        algorithm="PPO",
        policy_type="MlpPolicy",
        total_timesteps=2048,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_obs=False,
        normalize_reward=False,
        seed=0,
        device="cpu",
        verbose=0,
    )
    agent = train_target_agent_sb3(env, config=config)
    env.close()
    return agent


def _episode_returns(rollouts):
    return [sum(step["reward"] for step in traj) for traj in rollouts]


def test_mask_network_forward_and_range(target_agent: TargetAgent):
    """MaskNetwork should output a scalar in [0, 1] for a single observation."""
    env = gym.make("CartPole-v1")
    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=MaskTrainingConfig(hidden_sizes=(32, 32), device="cpu"),
    )
    obs, _ = env.reset(seed=0)
    score = mask_net.predict(obs)
    assert isinstance(score, (float, np.floating))
    assert 0.0 <= float(score) <= 1.0
    env.close()


def test_perturbed_policy_respects_mask_score(target_agent: TargetAgent):
    """PerturbedPolicy should return an action inside the action space."""
    env = gym.make("CartPole-v1")
    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=MaskTrainingConfig(hidden_sizes=(32, 32), device="cpu"),
    )
    perturbed = PerturbedPolicy(target_agent, mask_net, deterministic_target=True)

    obs, _ = env.reset(seed=1)
    action, _ = perturbed.predict(obs, deterministic=True)
    assert env.action_space.contains(action)

    # A deterministic target action should be reproducible when the mask is 1.0.
    target_action, _ = target_agent.predict(obs, deterministic=True)
    mask_score = mask_net.predict(obs)
    if mask_score > 0.99:
        assert int(action) == int(target_action)
    env.close()


def test_mask_training_increases_blinding(target_agent: TargetAgent):
    """Training the mask network should lower the average mask score (more blinding)."""
    env = gym.make("CartPole-v1")
    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=MaskTrainingConfig(hidden_sizes=(32, 32), device="cpu"),
    )

    # Pre-training mask scores.
    pre_rollouts = collect_masked_rollouts(
        env, target_agent, mask_net, n_episodes=10, alpha=0.0, deterministic_target=True
    )
    pre_scores = [step["mask_score"] for traj in pre_rollouts for step in traj]
    pre_mean = float(np.mean(pre_scores))

    # Train mask with a blinding coefficient large enough to drive scores down quickly.
    config = MaskTrainingConfig(
        alpha=1e-2,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        total_timesteps=1024,
        hidden_sizes=(32, 32),
        use_action=False,
        continuous_mask=True,
        device="cpu",
        seed=1,
        verbose=0,
    )
    trained_mask, _ = train_mask_network(env, target_agent, mask_net=mask_net, config=config)

    post_rollouts = collect_masked_rollouts(
        env, target_agent, trained_mask, n_episodes=10, alpha=0.0, deterministic_target=True
    )
    post_scores = [step["mask_score"] for traj in post_rollouts for step in traj]
    post_mean = float(np.mean(post_scores))

    assert post_mean < pre_mean, (
        f"Expected mask mean to decrease after training, but got "
        f"pre={pre_mean:.4f} -> post={post_mean:.4f}"
    )
    env.close()


def test_perturbed_policy_return_near_target(target_agent: TargetAgent):
    """After mask training, the perturbed-policy return should stay close to the target return."""
    env = gym.make("CartPole-v1")
    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=MaskTrainingConfig(hidden_sizes=(32, 32), device="cpu"),
    )

    config = MaskTrainingConfig(
        alpha=1e-2,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        total_timesteps=1024,
        hidden_sizes=(32, 32),
        use_action=False,
        continuous_mask=True,
        device="cpu",
        seed=2,
        verbose=0,
    )
    trained_mask, _ = train_mask_network(env, target_agent, mask_net=mask_net, config=config)

    target_rollouts = collect_masked_rollouts(
        env, target_agent, trained_mask, n_episodes=20, alpha=0.0, deterministic_target=True
    )
    # When alpha=0 the wrapper still uses the perturbed policy; the mask network is
    # trained to keep the perturbed-policy return high.
    perturbed_returns = _episode_returns(target_rollouts)

    # Evaluate the target policy alone for comparison.
    target_returns = []
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = target_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        target_returns.append(ep_return)

    mean_perturbed = float(np.mean(perturbed_returns))
    mean_target = float(np.mean(target_returns))
    # Tolerance: perturbed return should be within 20% or an absolute 20 points.
    assert mean_perturbed >= mean_target * 0.8 - 20.0, (
        f"Perturbed-policy return ({mean_perturbed:.2f}) dropped too far below "
        f"target return ({mean_target:.2f})"
    )
    env.close()


def test_extract_critical_states_ranking(target_agent: TargetAgent):
    """extract_critical_states should return states sorted by descending mask score."""
    env = gym.make("CartPole-v1")
    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=MaskTrainingConfig(hidden_sizes=(32, 32), device="cpu"),
    )
    rollouts = collect_masked_rollouts(
        env, target_agent, mask_net, n_episodes=5, alpha=0.0, deterministic_target=True
    )
    critical = extract_critical_states(rollouts, top_k=10)
    scores = [state["mask_score"] for state in critical]
    assert len(critical) <= 10
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    env.close()
