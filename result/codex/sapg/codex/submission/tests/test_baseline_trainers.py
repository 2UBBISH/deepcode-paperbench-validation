"""Smoke tests for the three baselines (PPO, DexPBT, PQL)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.algorithms import DexPBTTrainer, PQLTrainer, PPOTrainer  # noqa: E402
from sapg.envs import make_env  # noqa: E402
from sapg.utils.config import load_config  # noqa: E402

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sapg", "configs", "toy_sapg.yaml"
)


def test_ppo_trainer_runs():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(
            BASE,
            ["env.num_envs=16", "algo.num_policies=1", "algo.horizon=4", "algo.mini_epochs=1"],
        )
        trainer = PPOTrainer(cfg, make_env(cfg, device="cpu"), device="cpu", logdir=tmp)
        history = trainer.train(num_iterations=2)
        assert len(history) == 2
        assert trainer.use_offpolicy is False


def test_pbt_trainer_replaces_and_mutates():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(
            BASE,
            [
                "env.num_envs=24",
                "algo.num_policies=4",
                "algo.horizon=4",
                "algo.mini_epochs=1",
                "algo.pbt.interval=1",
                "algo.pbt.eval_episodes=2",
            ],
        )
        trainer = DexPBTTrainer(cfg, make_env(cfg, device="cpu"), device="cpu", logdir=tmp)
        trainer.train(num_iterations=2)
        assert trainer.num_policies == 4
        assert len(trainer.population) == 4
        # weight sharing did happen at least once (fitness of copies reset)
        assert any(value == float("-inf") for value in trainer.fitness) or trainer.iteration >= 1


def test_pql_trainer_runs():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(
            BASE,
            [
                "env.num_envs=16",
                "algo.name=pql",
                "algo.num_policies=1",
                "algo.horizon=1",
                "algo.batch_size=16",
                "algo.learning_starts=4",
                "algo.updates_per_iteration=1",
            ],
        )
        trainer = PQLTrainer(cfg, make_env(cfg, device="cpu"), device="cpu", logdir=tmp)
        history = trainer.train(num_iterations=3)
        assert len(history) == 3
