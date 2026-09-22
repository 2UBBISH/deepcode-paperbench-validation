#!/usr/bin/env python3
"""Montezuma's Revenge: pre-training, fine-tuning and evaluation (Section 3-5).

The three phases mirror the paper:

1. ``--phase pretrain``   -- train PPO+RND from Room 7 onward until the episode
   return reaches ~7000 (Section 3, Appendix B.2),
2. ``--phase collect``    -- collect 500 trajectories with the pre-trained agent
   to build the behavioral-cloning buffer,
3. ``--phase finetune``   -- fine-tune on the full game with the selected
   knowledge-retention method (``none``, ``bc``, ``ewc``).

Example::

    python scripts/train_montezuma.py --phase pretrain --steps 100000000
    python scripts/train_montezuma.py --phase finetune --method bc --steps 50000000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fpc.montezuma.config import MontezumaConfig
from fpc.montezuma.eval import room_success_rate, room_visitation
from fpc.montezuma.env import collect_pre_training_trajectories, make_env
from fpc.montezuma.model import AtariActorCritic
from fpc.montezuma.ppo import BehavioralCloningBuffer, PPORNDTrainer
from fpc.retention.base import RetentionConfig


class VectorisedEnv:
    """Minimal synchronous vectorised wrapper around the ALE environment."""

    def __init__(self, config: MontezumaConfig, num_envs: int, seed: int = 0) -> None:
        from fpc.montezuma.env import AtariPreprocessor

        self.config = config
        self.num_envs = num_envs
        self.num_actions = 18
        self.observation_shape = (config.state_stack_size, config.preproc_height, config.preproc_width)
        self.envs = [make_env(config, seed=seed + i) for i in range(num_envs)]
        self.preprocessors = [
            AtariPreprocessor(config.preproc_height, config.preproc_width, config.state_stack_size)
            for _ in range(num_envs)
        ]
        self._obs = None

    def reset(self):
        obs = []
        for env, pre in zip(self.envs, self.preprocessors):
            raw, _ = env.reset()
            obs.append(pre.reset(raw))
        self._obs = np.stack(obs)
        return self._obs, [{} for _ in self.envs]

    def step(self, actions):
        obs, rewards, dones = [], [], []
        for i, (env, pre) in enumerate(zip(self.envs, self.preprocessors)):
            raw, reward, terminated, truncated, _ = env.step(int(actions[i]))
            done = terminated or truncated
            if done:
                raw, _ = env.reset()
                obs.append(pre.reset(raw))
            else:
                obs.append(pre.step(raw))
            rewards.append(reward)
            dones.append(float(done))
        self._obs = np.stack(obs)
        return self._obs, np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=np.float32)

    def close(self) -> None:
        for env in self.envs:
            env.close()


def build_config(args) -> MontezumaConfig:
    config = MontezumaConfig()
    config.num_env = args.num_env
    config.total_steps = args.steps
    config.pretrain_room = args.room
    config.device = args.device
    config.seed = args.seed
    config.log_dir = args.log_dir
    config.save_dir = args.save_dir
    config.retention = RetentionConfig(
        method=args.method,
        coefficient=args.coefficient if args.coefficient is not None else (0.01 if args.method == "bc" else 1.0),
        decay=1.0,
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pretrain", "collect", "finetune", "eval"], default="finetune")
    parser.add_argument("--method", choices=["none", "bc", "ewc"], default="bc")
    parser.add_argument("--coefficient", type=float, default=None)
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--num-env", type=int, default=128)
    parser.add_argument("--room", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-dir", default="results/montezuma")
    parser.add_argument("--save-dir", default="results/montezuma/checkpoints")
    parser.add_argument("--init-from", default=None, help="checkpoint of pi_* to initialise fine-tuning")
    parser.add_argument("--bc-buffer", default=None, help=".npz with the collected BC trajectories")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    config = build_config(args)

    venv = VectorisedEnv(config, config.num_env, seed=args.seed)
    teacher, bc_buffer = None, None
    if args.init_from:
        state = torch.load(args.init_from, map_location=args.device)
        teacher = AtariActorCritic(state["num_actions"]).to(args.device)
        teacher.load_state_dict(state["model"])
    if args.bc_buffer:
        data = np.load(args.bc_buffer)
        bc_buffer = BehavioralCloningBuffer(data["observations"], data["actions"], device=args.device)

    trainer = PPORNDTrainer(config, venv, device=args.device, teacher=teacher, bc_buffer=bc_buffer)
    if args.init_from:
        trainer.model.load_state_dict(teacher.state_dict())

    if args.phase in ("pretrain", "finetune"):
        history = trainer.train(total_steps=config.total_steps)
        torch.save(
            {"model": trainer.model.state_dict(), "num_actions": venv.num_actions, "steps": trainer.global_step},
            os.path.join(args.save_dir, f"{args.phase}-{args.method}-{args.seed}.pt"),
        )
        import json

        with open(os.path.join(args.log_dir, f"{args.phase}-{args.method}-{args.seed}.json"), "w") as handle:
            json.dump(history, handle)

    elif args.phase == "collect":
        if teacher is None:
            raise SystemExit("--init-from is required to collect BC trajectories")
        data = collect_pre_training_trajectories(teacher, config, config.bc_trajectories, device=args.device)
        np.savez(os.path.join(args.log_dir, "bc_buffer.npz"), **data)

    else:  # eval
        success = room_success_rate(trainer.model, config, room=args.room, device=args.device)
        visitation = room_visitation(trainer.model, config, device=args.device)
        print(f"Room {args.room} success rate: {success:.3f}")
        print("Room visitation:", visitation)


if __name__ == "__main__":
    main()
