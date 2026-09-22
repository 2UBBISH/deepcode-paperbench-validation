"""SAPG: Split and Aggregate Policy Gradients -- main entry point.

Trains either SAPG or vanilla PPO on one of the supported manipulation tasks
(AllegroKuka Regrasping/Throw/Reorientation, ShadowHand, AllegroHand).

Usage
-----
    python main.py --config configs/allegro_kuka.yaml --task regrasping
    python main.py --config configs/shadow_hand.yaml --algo ppo
    python main.py --config configs/allegro_hand.yaml --num_envs 1024 --num_workers 4

The script is intentionally dependency-light: it only requires torch, numpy and
PyYAML.  The environments shipped with this repository are lightweight kinematic
proxies for the IsaacGym tasks described in the paper, so the full training loop
can be exercised on CPU as well as GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

# Make `sapg` importable when running `python main.py` from inside the package dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sapg.aggregation import AggregationMode, block_slices, split_worker_ids  # noqa: E402
from sapg.policy import build_policy  # noqa: E402
from sapg.ppo import PPO  # noqa: E402
from sapg.rollout_buffer import MultiWorkerRolloutBuffer  # noqa: E402
from sapg.sapg_algorithm import SAPG  # noqa: E402
from envs import make_env  # noqa: E402
from envs.curriculum import build_curriculum  # noqa: E402


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config, resolving the ``defaults: [default]`` inheritance chain."""
    cfg: Dict[str, Any] = {}
    if config_path is not None:
        raw = _load_yaml(config_path)
        defaults = raw.pop("defaults", None)
        if defaults:
            if isinstance(defaults, str):
                defaults = [defaults]
            base_dir = os.path.dirname(os.path.abspath(config_path))
            for entry in defaults:
                # Hydra-style entries may be plain names or dicts.
                name = entry if isinstance(entry, str) else list(entry.values())[0]
                base_path = os.path.join(base_dir, f"{name}.yaml")
                if os.path.exists(base_path):
                    cfg.update(_load_yaml(base_path))
        cfg.update(raw)
    else:
        default_path = os.path.join(_HERE, "configs", "default.yaml")
        if os.path.exists(default_path):
            cfg.update(_load_yaml(default_path))

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[main] CUDA requested but unavailable -- falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def _resolve_num_workers(cfg: Dict[str, Any], num_envs: int) -> int:
    num_workers = int(cfg.get("num_workers", 1) or 1)
    num_workers = max(1, min(num_workers, num_envs))
    return num_workers


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: Dict[str, Any], algo: str = "sapg") -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0) or 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(cfg)
    num_envs = int(cfg.get("num_envs", 64) or 64)
    horizon = int(cfg.get("horizon", 32) or 32)
    max_iterations = int(cfg.get("max_iterations", 100) or 100)
    num_workers = _resolve_num_workers(cfg, num_envs)

    env_name = str(cfg.get("env_name", "allegro_kuka"))
    task = str(cfg.get("task", "regrasping"))

    print(f"[main] algo={algo} env={env_name} task={task} num_envs={num_envs} "
          f"num_workers={num_workers} horizon={horizon} device={device}")

    env = make_env(
        env_name=env_name,
        num_envs=num_envs,
        task=task,
        cfg=cfg,
        device=device,
        seed=seed,
    )
    obs_dim = env.get_obs_dim()
    action_dim = env.get_action_dim()
    print(f"[main] obs_dim={obs_dim} action_dim={action_dim}")

    # ---- policy -----------------------------------------------------------
    policy = build_policy(cfg, obs_dim=obs_dim, action_dim=action_dim, num_workers=num_workers)

    # ---- algorithm --------------------------------------------------------
    if algo == "ppo":
        trainer = PPO(cfg=cfg, policy=policy, obs_dim=obs_dim, action_dim=action_dim,
                      num_envs=num_envs, device=device)
        trainer.init()
        worker_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        buffer = MultiWorkerRolloutBuffer(
            num_workers=1,
            horizon=horizon,
            envs_per_worker=num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            recurrent=bool(cfg.get("use_lstm", False)),
            hidden_size=int(cfg.get("lstm_hidden", 768) or 768),
            gamma=float(cfg.get("gamma", 0.99)),
            tau=float(cfg.get("tau", 0.95)),
        )
    else:
        trainer = SAPG(cfg=cfg, policy=policy, obs_dim=obs_dim, action_dim=action_dim,
                       num_envs=num_envs, num_workers=num_workers, device=device)
        worker_ids = trainer.env_worker_ids
        buffer = MultiWorkerRolloutBuffer(
            num_workers=num_workers,
            horizon=horizon,
            envs_per_worker=num_envs // num_workers,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            recurrent=bool(cfg.get("use_lstm", False)),
            hidden_size=int(cfg.get("lstm_hidden", 768) or 768),
            gamma=float(cfg.get("gamma", 0.99)),
            tau=float(cfg.get("tau", 0.95)),
        )

    curriculum = build_curriculum(cfg)

    # ---- rollout state ----------------------------------------------------
    obs = env.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    hidden_state = trainer.init_hidden(num_envs) if trainer.is_recurrent else None
    masks = torch.ones(num_envs, dtype=torch.float32, device=device)

    output_dir = str(cfg.get("output_dir", "runs"))
    os.makedirs(output_dir, exist_ok=True)
    log_interval = int(cfg.get("log_interval", 10) or 10)
    save_interval = int(cfg.get("save_interval", 100) or 100)

    history = {"iteration": [], "mean_reward": [], "mean_successes": [], "tolerance": []}
    start_time = time.time()
    global_step = 0

    for it in range(max_iterations):
        buffer.reset()
        ep_rewards = np.zeros(num_envs, dtype=np.float64)
        ep_successes = np.zeros(num_envs, dtype=np.float64)
        completed_rewards = []
        completed_successes = []

        for _t in range(horizon):
            with torch.no_grad():
                actions, log_probs, values, new_hidden = trainer.act(
                    obs_t, worker_ids, hidden_state, masks, deterministic=False
                )
            actions_np = actions.detach().cpu().numpy()
            next_obs, rewards, dones, infos = env.step(actions_np)

            buffer.add(
                obs=obs_t,
                actions=actions,
                log_probs=log_probs,
                rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
                dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
                values=values,
                worker_ids=worker_ids,
                hidden_state=hidden_state,
                masks=masks,
            )

            ep_rewards += rewards
            ep_successes += np.asarray(infos.get("successes", np.zeros(num_envs)), dtype=np.float64)

            done_mask = np.asarray(dones, dtype=bool)
            if done_mask.any():
                completed_rewards.extend(ep_rewards[done_mask].tolist())
                completed_successes.extend(ep_successes[done_mask].tolist())
                ep_rewards[done_mask] = 0.0
                ep_successes[done_mask] = 0.0

            masks = torch.as_tensor(1.0 - np.asarray(dones, dtype=np.float32), device=device)
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            hidden_state = new_hidden
            global_step += num_envs

        # ---- bootstrap values --------------------------------------------
        with torch.no_grad():
            _, _, last_values, _ = trainer.act(obs_t, worker_ids, hidden_state, masks,
                                               deterministic=True)
        buffer.compute_returns_and_advantages(last_values)

        # ---- update -------------------------------------------------------
        stats = trainer.update(buffer)

        # ---- curriculum ---------------------------------------------------
        if curriculum is not None and completed_successes:
            mean_succ = float(np.mean(completed_successes))
            curriculum.update(mean_succ)
            if hasattr(env, "curriculum") and env.curriculum is not None:
                env.curriculum.load_state_dict(curriculum.state_dict())

        mean_reward = float(np.mean(completed_rewards)) if completed_rewards else float("nan")
        mean_succ = float(np.mean(completed_successes)) if completed_successes else float("nan")
        history["iteration"].append(it)
        history["mean_reward"].append(mean_reward)
        history["mean_successes"].append(mean_succ)
        history["tolerance"].append(curriculum.get_tolerance() if curriculum else float("nan"))

        if it % log_interval == 0 or it == max_iterations - 1:
            elapsed = time.time() - start_time
            print(
                f"[iter {it:5d}] steps={global_step:9d} "
                f"reward={mean_reward:8.3f} successes={mean_succ:6.3f} "
                f"tol={history['tolerance'][-1]:.4f} "
                f"pi_loss={stats.get('policy_loss', float('nan')):8.4f} "
                f"v_loss={stats.get('value_loss', float('nan')):8.4f} "
                f"kl={stats.get('approx_kl', float('nan')):7.5f} "
                f"lr={stats.get('learning_rate', float('nan')):.2e} "
                f"t={elapsed:6.1f}s"
            )

        if save_interval > 0 and (it + 1) % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_{it + 1}.pt")
            torch.save(
                {
                    "iteration": it,
                    "policy": trainer.state_dict(),
                    "env": env.get_state_dict() if hasattr(env, "get_state_dict") else {},
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"[main] saved checkpoint -> {ckpt_path}")

    # ---- final artifacts --------------------------------------------------
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    final_ckpt = os.path.join(output_dir, "checkpoint_final.pt")
    torch.save(
        {
            "iteration": max_iterations,
            "policy": trainer.state_dict(),
            "env": env.get_state_dict() if hasattr(env, "get_state_dict") else {},
            "config": cfg,
        },
        final_ckpt,
    )
    print(f"[main] training complete. artifacts in {output_dir}")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train SAPG or PPO on manipulation tasks.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config (e.g. configs/allegro_kuka.yaml).")
    p.add_argument("--algo", type=str, default="sapg", choices=["sapg", "ppo"],
                   help="Which algorithm to run.")
    p.add_argument("--env_name", type=str, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--max_iterations", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--aggregation_mode", type=str, default=None,
                   choices=["leader", "symmetric", "none"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {
        "env_name": args.env_name,
        "task": args.task,
        "num_envs": args.num_envs,
        "num_workers": args.num_workers,
        "horizon": args.horizon,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "device": args.device,
        "output_dir": args.output_dir,
        "aggregation_mode": args.aggregation_mode,
    }
    cfg = load_config(args.config, overrides)
    train(cfg, algo=args.algo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
