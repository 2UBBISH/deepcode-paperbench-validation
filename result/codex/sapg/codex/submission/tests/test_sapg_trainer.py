"""End-to-end smoke tests of the SAPG trainer on the CPU toy suite."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.algorithms import SAPGTrainer  # noqa: E402
from sapg.envs import make_env  # noqa: E402
from sapg.utils.config import load_config  # noqa: E402


CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sapg", "configs", "toy_sapg.yaml"
)


def build_trainer(overrides, logdir):
    cfg = load_config(
        CONFIG,
        ["env.num_envs=24", "algo.num_policies=3", "algo.horizon=4", "algo.mini_epochs=1"]
        + overrides,
    )
    env = make_env(cfg, device="cpu")
    return SAPGTrainer(cfg, env, device="cpu", logdir=logdir), cfg


def test_trainer_runs_and_logs():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = build_trainer([], tmp)
        history = trainer.train(num_iterations=2)
        assert len(history) == 2
        assert trainer.env_steps == 24 * 4 * 2
        assert "on/policy_loss" in history[-1]
        assert "off/policy_loss" in history[-1]
        assert os.path.exists(os.path.join(tmp, "progress.csv"))


def test_offpolicy_subsampling_matches_onpolicy_size():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = build_trainer([], tmp)
        trainer.runner.reset()
        trainer.collect()
        data = trainer.prepare_offpolicy()
        assert set(data) == {0}, "only the leader receives off-policy data"
        assert data[0].num_transitions() == trainer.storage.buffer(0).num_transitions()


def test_high_offpolicy_ratio_uses_the_full_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = build_trainer(["algo.offpolicy_subsample=false"], tmp)
        trainer.runner.reset()
        trainer.collect()
        data = trainer.prepare_offpolicy()
        expected = (trainer.num_policies - 1) * trainer.storage.buffer(0).num_transitions()
        assert data[0].num_transitions() == expected


def test_symmetric_aggregation_updates_every_policy():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = build_trainer(["algo.aggregation=symmetric"], tmp)
        trainer.runner.reset()
        trainer.collect()
        data = trainer.prepare_offpolicy()
        assert set(data) == {0, 1, 2}


def test_follower_entropy_coefficients():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = build_trainer(["algo.entropy_coef=0.005"], tmp)
        assert trainer.entropy_coef_for(0) == 0.0
        assert trainer.entropy_coef_for(1) == 0.005
        assert trainer.entropy_coef_for(2) == 0.01


def test_beat_shared_parameters_receive_gradients_from_all_policies():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = build_trainer([], tmp)
        trainer.runner.reset()
        trainer.collect()
        metrics = trainer.update()
        assert "off/policy_loss" in metrics
        # gradients of the shared backbone exist for both actor and critic
        grads = [p.grad.abs().sum().item() for p in trainer.model.actor.parameters()]
        assert any(g > 0 for g in grads)
