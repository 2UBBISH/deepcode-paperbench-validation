"""End-to-end smoke test for the FRE implementation.

Runs a handful of encoder and policy steps on a synthetic offline dataset to
verify that the whole pipeline (reward priors -> batched reward functions ->
discretisation -> transformer VAE encoder -> decoder -> IQL-C) is wired up
correctly and that shapes match.  This is deliberately tiny: it exercises code
paths rather than producing meaningful numbers.

Run with ``python -m tests.test_fre_smoke`` or ``pytest tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fre.configs import TrainConfig
from fre.datasets import OfflineDataset, from_trajectories
from fre.fre import FRE, FREConfig
from fre.reward_functions import build_prior, discretize_reward, PRIOR_REGISTRY
from fre.tasks.antmaze import (
    AntMazeStatePreprocessor,
    make_antmaze_goal_tasks,
    make_antmaze_directional_tasks,
    make_antmaze_simplex_tasks,
    make_antmaze_path_tasks,
    ANTMAZE_OBS_DIM,
)
from fre.training import FRETrainer


def make_synthetic_antmaze_dataset(num_traj: int = 40, traj_len: int = 60) -> OfflineDataset:
    rng = np.random.default_rng(0)
    obs_dim, act_dim = ANTMAZE_OBS_DIM, 8
    trajectories, actions = [], []
    for _ in range(num_traj):
        xy = rng.uniform(0.0, 30.0, size=(traj_len, 2))
        rest = rng.normal(size=(traj_len, obs_dim - 2)) * 0.3
        obs = np.concatenate([xy, rest], axis=-1).astype(np.float32)
        trajectories.append(obs)
        actions.append(rng.uniform(-1, 1, size=(traj_len, act_dim)).astype(np.float32))
    return from_trajectories(trajectories, actions)


def test_dataset_goal_sampling():
    dataset = make_synthetic_antmaze_dataset()
    goals = dataset.sample_goals(64)
    assert goals.shape == (64, ANTMAZE_OBS_DIM)
    states = dataset.random_states(128)
    assert states.shape == (128, ANTMAZE_OBS_DIM)
    batch = dataset.sample_transitions(32)
    assert batch["observations"].shape == (32, ANTMAZE_OBS_DIM)
    assert batch["actions"].shape[0] == 32


def test_prior_families_shapes():
    dataset = make_synthetic_antmaze_dataset()
    sampler = dataset.goal_sampler(np.random.default_rng(0))
    for name in list(PRIOR_REGISTRY) + ["FRE-hint"]:
        kwargs = {}
        if name == "FRE-hint":
            from fre.tasks.exorl import build_antmaze_hint_priors

            kwargs["hint_priors"] = build_antmaze_hint_priors()
        prior = build_prior(name, dataset.encoder_obs_dim, sampler, **kwargs)
        eta = prior.sample(16, torch.device("cpu"))
        enc = dataset.random_states(16 * 32).reshape(16, 32, -1)
        enc = eta.maybe_insert_goal_state(enc)
        r = eta.reward(enc)
        assert r.shape == (16, 32), (name, r.shape)
        flat = dataset.random_states(16)
        r_flat = eta.reward(flat)
        assert r_flat.shape == (16,), (name, r_flat.shape)
        bins = discretize_reward(r, 32, r_min=-1.0, r_max=1.0)
        assert bins.shape == (16, 32) and bins.min() >= 0 and bins.max() < 32


def test_encoder_decoder_forward():
    dataset = make_synthetic_antmaze_dataset()
    model = FRE(FREConfig(state_dim=dataset.encoder_obs_dim))
    states = torch.randn(4, 32, dataset.encoder_obs_dim)
    bins = torch.randint(0, 32, (4, 32))
    z = model.encode(states, bins)
    assert z.shape == (4, 128)
    dec_states = torch.randn(4, 8, dataset.encoder_obs_dim)
    pred = model.decode(dec_states, z)
    assert pred.shape == (4, 8)
    loss, recon, kl = model.loss(states, bins, dec_states, torch.randn(4, 8), return_parts=True)
    assert torch.isfinite(loss) and torch.isfinite(recon) and torch.isfinite(kl)


def test_trainer_runs_a_few_steps():
    dataset = make_synthetic_antmaze_dataset(num_traj=12, traj_len=40)
    config = TrainConfig(
        domain="antmaze",
        batch_size=8,
        num_encode_pairs=16,
        num_decode_pairs=4,
        encoder_steps=3,
        policy_steps=3,
        checkpoint_interval=100,
        log_interval=1,
        device="cpu",
        output_dir="/tmp/fre_smoke_runs",
        discretize_xy=True,
    )
    preprocess = AntMazeStatePreprocessor(num_bins=config.num_xy_bins)
    prior = build_prior(
        config.prior_name,
        dataset.encoder_obs_dim,
        dataset.goal_sampler(np.random.default_rng(1)),
    )
    trainer = FRETrainer(config, dataset, prior, preprocess)
    logs = []
    trainer.fit(log_fn=logs.append)
    assert len(logs) >= 2
    # A latent can be produced from an arbitrary downstream context set, and
    # the policy can act on it.
    task_states = dataset.random_states(32)
    task = make_antmaze_goal_tasks()[0]
    eta = prior.sample(1, torch.device("cpu"))
    z = trainer.encode_context(task_states[None], eta)
    assert z.shape == (1, 128)
    obs = torch.as_tensor(preprocess(dataset.observations[:2]))
    action = trainer.agent.act(obs, z.expand(2, -1))
    assert action.shape == (2, dataset.action_dim)


def test_task_rewards():
    rng = np.random.default_rng(0)
    obs = np.zeros((5, ANTMAZE_OBS_DIM), dtype=np.float32)
    obs[:, 0] = [28.0, 0.0, 35.0, 12.0, 33.0]
    obs[:, 1] = [0.0, 15.0, 24.0, 24.0, 16.0]
    tasks = make_antmaze_goal_tasks()
    rewards = np.stack([t.reward(obs) for t in tasks], axis=0)
    # Each position is the goal of exactly one task, so reward 0 on the diagonal.
    assert np.allclose(np.diag(rewards), 0.0)
    directional = make_antmaze_directional_tasks()
    obs_vel = np.zeros((1, ANTMAZE_OBS_DIM), dtype=np.float32)
    obs_vel[0, 15], obs_vel[0, 16] = 10.0, 0.0
    right = [t for t in directional if t.name == "vel_right"][0]
    assert right.reward(obs_vel)[0] == 1.0
    simplex = make_antmaze_simplex_tasks()
    assert len(simplex) == 5
    paths = make_antmaze_path_tasks()
    assert len(paths) == 3
    for t in simplex + paths:
        r = t.reward(np.random.default_rng(0).normal(size=(7, ANTMAZE_OBS_DIM)).astype(np.float32))
        assert r.shape == (7,)


def test_exorl_physics_features_and_tasks():
    from fre.tasks.exorl import (
        augment_with_physics,
        cheetah_physics_features,
        make_exorl_suites,
        walker_physics_features,
    )

    physics = np.zeros((5, 18), dtype=np.float64)
    physics[:, 1] = 1.0  # torso height
    physics[:, 9] = 2.0  # horizontal velocity
    walker_feats = walker_physics_features(physics)
    assert walker_feats.shape == (5, 3)
    assert np.allclose(walker_feats[:, 0], 2.0)
    assert np.allclose(walker_feats[:, 1], 1.0)  # cos(0) = 1
    cheetah_feats = cheetah_physics_features(physics)
    assert cheetah_feats.shape == (5, 1)

    obs = np.zeros((5, 24), dtype=np.float32)
    augmented = augment_with_physics("walker", obs, physics)
    assert augmented.shape == (5, 27)

    suites = make_exorl_suites("walker", goals=np.zeros((5, 24), dtype=np.float32), obs_std=np.ones(24))
    names = [s.name for s in suites]
    assert "exorl-walker-velocity" in names and "exorl-walker-goals" in names
    velocity_suite = suites[0]
    assert len(velocity_suite.tasks) == 4
    # A walker at 8 units/step should saturate the threshold-8 task only.
    aug = np.zeros((1, 27), dtype=np.float32)
    aug[0, 24] = 8.0
    scores = {t.name: float(t.reward(aug)[0]) for t in velocity_suite.tasks}
    assert scores["walker-vel-8"] == 1.0
    assert scores["walker-vel-0.1"] == 1.0


def test_gc_iql_smoke():
    from baselines.gc_iql import GCIQLAgent, GCIQLConfig, GoalRelabeler

    dataset = make_synthetic_antmaze_dataset(num_traj=10, traj_len=40)
    config = GCIQLConfig(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim)
    agent = GCIQLAgent(config, device=torch.device("cpu"))
    relabeler = GoalRelabeler(dataset, config, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        batch = dataset.sample_transitions(16, rng=rng)
        idx = rng.integers(0, len(dataset), size=16)
        goals, rewards, masks = relabeler.sample(idx, np.minimum(idx + 1, len(dataset) - 1))
        assert goals.shape == (16, dataset.obs_dim)
        assert set(np.unique(rewards)).issubset({-1.0, 0.0})
        agent.update(batch, goals, rewards, masks)
    action = agent.act(dataset.observations[:2], dataset.observations[0])
    assert action.shape == (2, dataset.action_dim)


def test_gc_bc_smoke():
    from baselines.gc_bc import GCBCAgent, GCBCConfig, GeometricGoalSampler

    dataset = make_synthetic_antmaze_dataset(num_traj=10, traj_len=40)
    config = GCBCConfig(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim)
    agent = GCBCAgent(config, device=torch.device("cpu"))
    sampler = GeometricGoalSampler(dataset, geometric_p=config.geometric_p, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        idx = rng.integers(0, len(dataset), size=16)
        goal = sampler.sample(idx)
        agent.update(
            torch.as_tensor(dataset.observations[idx]),
            torch.as_tensor(goal),
            torch.as_tensor(dataset.actions[idx]),
        )
    # GC-BC must only sample *future* goals (never the current state).
    idx = np.arange(0, 100)
    goal = sampler.sample(idx)
    assert not np.allclose(goal, dataset.observations[idx])
    action = agent.act(dataset.observations[:1], goal[:1])
    assert action.shape == (1, dataset.action_dim)


def test_opal_smoke():
    from baselines.opal import OPALAgent, OPALConfig, sample_trajectory_chunks

    dataset = make_synthetic_antmaze_dataset(num_traj=10, traj_len=40)
    config = OPALConfig(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim, context_length=8)
    agent = OPALAgent(config, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    for _ in range(2):
        chunk = sample_trajectory_chunks(dataset, 4, 8, rng)
        stats = agent.update_vae(chunk["states"], chunk["actions"], chunk["next_states"])
        assert "vae_loss" in stats
    skills = agent.sample_skills(10, rng)
    assert skills.shape == (10, config.skill_dim)
    action = agent.act(dataset.observations[:1], skills[0])
    assert action.shape == (1, dataset.action_dim)


def test_evaluation_pipeline_with_stub_env():
    """Exercise the evaluation driver end-to-end against a stub environment."""
    from fre.configs import EvalConfig
    from fre.evaluate import FREPolicy, evaluate_task
    from fre.iql import IQLAgent, IQLConfig
    from fre.tasks.antmaze import make_antmaze_goal_tasks

    class StubEnv:
        """Moves the ant straight to the goal, so the task reward is optimal."""

        def __init__(self, goal):
            self.goal = np.asarray(goal, dtype=np.float32)
            self.steps = 0

        def reset(self, seed=None):
            self.steps = 0
            obs = np.zeros(ANTMAZE_OBS_DIM, dtype=np.float32)
            return obs

        def step(self, action):
            self.steps += 1
            obs = np.zeros(ANTMAZE_OBS_DIM, dtype=np.float32)
            obs[:2] = self.goal
            return obs, 0.0, True, {}

    dataset = make_synthetic_antmaze_dataset(num_traj=8, traj_len=30)
    config = TrainConfig(domain="antmaze", device="cpu", batch_size=4)
    model = FRE(FREConfig(state_dim=dataset.encoder_obs_dim))
    agent = IQLAgent(IQLConfig(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim), device=torch.device("cpu"))
    policy = FREPolicy(model, agent, AntMazeStatePreprocessor(), torch.device("cpu"))
    task = make_antmaze_goal_tasks(max_episode_steps=5)[0]
    env = StubEnv(task.goal)
    result = evaluate_task(env, task, policy, dataset, EvalConfig(num_episodes=2, num_encode_pairs=8), seed=0)
    assert result["raw_return_mean"] == 0.0  # reached immediately => reward 0
    assert result["normalized_mean"] > 0.0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                import traceback

                print(f"FAIL {name}: {exc}")
                traceback.print_exc()
    print("failures:", failures)
    sys.exit(1 if failures else 0)
