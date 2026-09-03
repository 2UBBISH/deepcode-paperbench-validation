"""
Ablation: prior-mixture scaling (Figure 5).

Trains the FRE reward-prior encoder/decoder and the FRE-conditioned IQL agent
using different subsets of the random reward-function prior:

    goals  -> singleton goal-reaching rewards only
    linear -> sparse random-linear rewards only
    mlp    -> random two-layer MLP rewards only
    all    -> uniform mixture of all three families

The script is self-contained and does not require the phased training entry
points; it reuses the same modules as the main experiments so the comparison is
faithful.  After training each variant it runs the standard 32-example
zero-shot evaluation protocol for the requested domain and writes a JSON report
with per-task and aggregate scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Allow running from anywhere while still importing the repository packages.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fre.agent import FREAgent  # noqa: E402
from fre.dataset import build_state_pool, load_offline_dataset, make_synthetic_dataset  # noqa: E402
from fre.fre_vae import FREVAE  # noqa: E402
from fre.reward_prior import (  # noqa: E402
    LinearReward,
    MLPReward,
    SingletonGoalReward,
)
from fre.utils import get_logger, resolve_device, set_seed  # noqa: E402

from envs.antmaze_wrapper import (  # noqa: E402
    ANTMAZE_TASKS,
    evaluate_antmaze_policy,
    make_antmaze_task_reward,
)
from envs.exorl_wrapper import (  # noqa: E402
    EXORL_TASKS,
    evaluate_exorl_policy,
    make_exorl_task_reward,
)
from envs.kitchen_wrapper import (  # noqa: E402
    KITCHEN_TASKS,
    evaluate_kitchen_policy,
    make_kitchen_task_reward,
)

LOGGER = get_logger("ablation_prior_mixture")

# Maps domain names to the downstream tasks used for zero-shot evaluation.
DOMAIN_TASKS = {
    "antmaze": ANTMAZE_TASKS,
    "kitchen": KITCHEN_TASKS,
    "walker": EXORL_TASKS,
    "cheetah": EXORL_TASKS,
}

MIXTURE_FAMILIES = {
    "goals": ("goal",),
    "linear": ("linear",),
    "mlp": ("mlp",),
    "all": ("goal", "linear", "mlp"),
}


class AblationRewardPrior:
    """Deterministic subset of the FRE reward-prior distribution.

    The production :class:`fre.reward_prior.RewardPrior` always samples the
    three reward families uniformly.  For the Figure 5 ablation we need to
    restrict sampling to a chosen subset while preserving the exact same reward
    family definitions.
    """

    def __init__(
        self,
        state_dim: int,
        state_pool: np.ndarray,
        families: Sequence[str] = ("goal", "linear", "mlp"),
        goal_epsilon: float = 1.0,
        p_mask: float = 0.75,
        mlp_hidden: int = 256,
        device: torch.device | str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.state_dim = state_dim
        self.device = torch.device(device)
        self.families = tuple(families)
        if not self.families:
            raise ValueError("At least one reward family must be selected")
        self.goal_epsilon = goal_epsilon
        self.p_mask = p_mask
        self.mlp_hidden = mlp_hidden

        self._state_pool = torch.as_tensor(
            np.asarray(state_pool, dtype=np.float32), device=self.device
        )
        if self._state_pool.ndim == 1:
            self._state_pool = self._state_pool.unsqueeze(0)
        self._rng = np.random.RandomState(seed)

    def _sample_goal_reward(self) -> SingletonGoalReward:
        idx = int(self._rng.randint(0, len(self._state_pool)))
        goal = self._state_pool[idx]
        return SingletonGoalReward(goal=goal, epsilon=self.goal_epsilon)

    def _sample_linear_reward(self) -> LinearReward:
        weights = torch.empty(self.state_dim, device=self.device).uniform_(-1.0, 1.0)
        mask = torch.from_numpy(
            (self._rng.uniform(0.0, 1.0, size=(self.state_dim,)) < self.p_mask)
        ).float().to(self.device)
        return LinearReward(weights=weights, mask=mask)

    def _sample_mlp_reward(self) -> MLPReward:
        net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.mlp_hidden, self.mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.mlp_hidden, 1),
        ).to(self.device)
        net.eval()
        for param in net.parameters():
            param.requires_grad_(False)
        return MLPReward(net=net)

    def sample_reward_fn(self):
        kind = self.families[int(self._rng.randint(0, len(self.families)))]
        if kind == "goal":
            return self._sample_goal_reward()
        if kind == "linear":
            return self._sample_linear_reward()
        return self._sample_mlp_reward()

    def sample_reward_fns(self, batch_size: int) -> List[object]:
        return [self.sample_reward_fn() for _ in range(batch_size)]

    def evaluate(self, reward_fn: Callable[[torch.Tensor], torch.Tensor], states: torch.Tensor):
        return reward_fn(states)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FRE with reward-prior subsets and evaluate zero-shot (Figure 5 ablation)."
    )
    parser.add_argument("--domain", type=str, default="antmaze",
                        choices=["antmaze", "kitchen", "walker", "cheetah", "synthetic"])
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--mixtures", type=str, default="goals,linear,mlp,all",
                        help="Comma-separated prior-mixture variants to train.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="ablation_results")

    # Phase 1 (FRE VAE)
    parser.add_argument("--vae-steps", type=int, default=20000)
    parser.add_argument("--vae-batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--reward-bins", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--encoder-states", type=int, default=32)
    parser.add_argument("--decoder-states", type=int, default=256)
    parser.add_argument("--vae-lr", type=float, default=1e-4)
    parser.add_argument("--vae-beta", type=float, default=1.0)

    # Phase 2 (IQL RL)
    parser.add_argument("--rl-steps", type=int, default=20000)
    parser.add_argument("--rl-batch-size", type=int, default=256)
    parser.add_argument("--rl-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--awr-temperature", type=float, default=3.0)
    parser.add_argument("--target-tau", type=float, default=0.005)

    # Evaluation
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--state-pool-size", type=int, default=None)
    return parser.parse_args(argv)


def load_dataset_and_pool(args: argparse.Namespace, device: torch.device):
    if args.domain == "synthetic":
        dataset = make_synthetic_dataset(state_dim=17, action_dim=8, size=20000, seed=args.seed)
        state_dim = dataset.states.shape[1]
        action_dim = dataset.actions.shape[1]
    else:
        dataset = load_offline_dataset(
            args.domain, dataset_name=args.dataset_name, dataset_path=args.dataset_path
        )
        state_dim = getattr(dataset, "state_dim", None) or int(dataset.states.shape[1])
        action_dim = getattr(dataset, "action_dim", None) or int(dataset.actions.shape[1])

    state_pool = build_state_pool(dataset, max_pool_size=args.state_pool_size)
    return dataset, state_pool, state_dim, action_dim


def train_vae(
    args: argparse.Namespace,
    prior: AblationRewardPrior,
    state_pool: np.ndarray,
    state_dim: int,
    device: torch.device,
    mixture_name: str,
    output_dir: Path,
) -> FREVAE:
    LOGGER.info("[%s] Training FRE VAE for %d steps", mixture_name, args.vae_steps)
    vae = FREVAE(
        state_dim=state_dim,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        reward_bins=args.reward_bins,
        embedding_dim=args.embedding_dim,
        decoder_hidden=tuple(args.decoder_hidden),
        beta=args.vae_beta,
        device=device,
    ).to(device)

    optimizer = vae.configure_optimizer(lr=args.vae_lr)
    pool_tensor = torch.as_tensor(state_pool, dtype=torch.float32, device=device)
    n_pool = pool_tensor.shape[0]

    for step in range(1, args.vae_steps + 1):
        reward_fns = prior.sample_reward_fns(args.vae_batch_size)
        enc_idx = torch.randint(0, n_pool, (args.vae_batch_size, args.encoder_states), device=device)
        dec_idx = torch.randint(0, n_pool, (args.vae_batch_size, args.decoder_states), device=device)
        enc_states = pool_tensor[enc_idx]
        dec_states = pool_tensor[dec_idx]

        enc_rewards = torch.stack([fn(enc_states[i]) for i, fn in enumerate(reward_fns)])
        dec_rewards = torch.stack([fn(dec_states[i]) for i, fn in enumerate(reward_fns)])

        metrics = vae.training_step(
            enc_states, enc_rewards, dec_states, dec_rewards, optimizer
        )

        if step % args.log_interval == 0 or step == 1:
            recon = float(metrics.get("recon_loss", metrics.get("reconstruction_loss", 0.0)))
            kl = float(metrics.get("kl_loss", 0.0))
            total = float(metrics.get("total_loss", metrics.get("loss", recon + kl)))
            LOGGER.info(
                "[%s] VAE step %d/%d recon=%.5f kl=%.5f total=%.5f",
                mixture_name, step, args.vae_steps, recon, kl, total,
            )

    # Freeze the encoder/decoder for phase 2.
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    checkpoint_path = output_dir / f"{mixture_name}_vae.pt"
    torch.save(
        {
            "model_state_dict": vae.state_dict(),
            "state_dim": state_dim,
            "latent_dim": args.latent_dim,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "reward_bins": args.reward_bins,
            "embedding_dim": args.embedding_dim,
        },
        checkpoint_path,
    )
    LOGGER.info("[%s] Saved VAE checkpoint to %s", mixture_name, checkpoint_path)
    return vae


def train_agent(
    args: argparse.Namespace,
    vae: FREVAE,
    prior: AblationRewardPrior,
    dataset,
    state_pool: np.ndarray,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    mixture_name: str,
    output_dir: Path,
) -> FREAgent:
    LOGGER.info("[%s] Training FRE-conditioned IQL agent for %d steps", mixture_name, args.rl_steps)
    agent = FREAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        vae=vae,
        reward_prior=prior,
        state_pool=state_pool,
        dataset=dataset,
        freeze_vae=True,
        gamma=args.gamma,
        expectile=args.expectile,
        awr_temperature=args.awr_temperature,
        target_tau=args.target_tau,
        lr=args.rl_lr,
        device=device,
    )
    agent.to(device)

    for step in range(1, args.rl_steps + 1):
        metrics = agent.train_on_dataset(batch_size=args.rl_batch_size)
        if step % args.log_interval == 0 or step == 1:
            q_loss = float(metrics.get("q_loss", 0.0))
            v_loss = float(metrics.get("v_loss", 0.0))
            pi_loss = float(metrics.get("policy_loss", 0.0))
            LOGGER.info(
                "[%s] RL step %d/%d q=%.5f v=%.5f pi=%.5f",
                mixture_name, step, args.rl_steps, q_loss, v_loss, pi_loss,
            )

    checkpoint_path = output_dir / f"{mixture_name}_agent.pt"
    agent.save(str(checkpoint_path))
    LOGGER.info("[%s] Saved agent checkpoint to %s", mixture_name, checkpoint_path)
    return agent


def make_task_reward(domain: str, task_name: str):
    if domain == "antmaze":
        return make_antmaze_task_reward(task_name)
    if domain == "kitchen":
        return make_kitchen_task_reward(task_name)
    return make_exorl_task_reward(task_name)


def evaluate_task(domain: str, task_name: str, policy_fn, num_episodes: int, seed: int):
    if domain == "antmaze":
        result = evaluate_antmaze_policy(policy_fn, task_name, num_episodes=num_episodes, seed=seed)
        return result.get("normalized_score", result.get("score", 0.0))
    if domain == "kitchen":
        result = evaluate_kitchen_policy(policy_fn, task_name, num_episodes=num_episodes, seed=seed)
        return result.get("normalized_score", result.get("score", 0.0))
    result = evaluate_exorl_policy(
        policy_fn, task_name, domain=domain, num_episodes=num_episodes, seed=seed
    )
    return result.get("normalized_score", result.get("score", 0.0))


def evaluate_agent(
    agent: FREAgent,
    domain: str,
    tasks: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    agent.eval() if hasattr(agent, "eval") else None
    scores: Dict[str, float] = {}
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed
    for task_name in tasks:
        reward_fn = make_task_reward(domain, task_name)
        z = agent.encode_task(reward_fn, num_examples=args.num_examples)
        z = z.to(device)

        def policy_fn(obs: np.ndarray, z: torch.Tensor = z):
            return agent.get_action(obs, z=z, deterministic=True)

        score = evaluate_task(domain, task_name, policy_fn, args.num_episodes, eval_seed)
        scores[task_name] = float(score)
        LOGGER.info("[%s] %s/%s score=%.2f", domain, domain, task_name, score)
    if scores:
        scores["mean"] = float(np.mean(list(scores.values())))
    return scores


def run_mixture(
    mixture_name: str,
    families: Sequence[str],
    args: argparse.Namespace,
    dataset,
    state_pool: np.ndarray,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    set_seed(args.seed)
    prior = AblationRewardPrior(
        state_dim=state_dim,
        state_pool=state_pool,
        families=families,
        device=device,
        seed=args.seed,
    )
    vae = train_vae(args, prior, state_pool, state_dim, device, mixture_name, output_dir)
    agent = train_agent(
        args, vae, prior, dataset, state_pool, state_dim, action_dim, device,
        mixture_name, output_dir,
    )
    if args.skip_eval:
        return {"mixture": mixture_name, "families": list(families)}

    tasks = DOMAIN_TASKS.get(args.domain, list(DOMAIN_TASKS.values())[0])
    scores = evaluate_agent(agent, args.domain, tasks, args, device)
    scores["mixture"] = mixture_name
    scores["families"] = list(families)
    return scores


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    set_seed(args.seed)

    dataset, state_pool, state_dim, action_dim = load_dataset_and_pool(args, device)
    LOGGER.info("Domain=%s state_dim=%d action_dim=%d state_pool=%s",
                args.domain, state_dim, action_dim, state_pool.shape)

    output_dir = Path(args.output_dir) / args.domain / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [m.strip() for m in args.mixtures.split(",") if m.strip()]
    results: List[Dict[str, float]] = []
    for mixture_name in requested:
        if mixture_name not in MIXTURE_FAMILIES:
            LOGGER.warning("Unknown mixture '%s'; skipping", mixture_name)
            continue
        families = MIXTURE_FAMILIES[mixture_name]
        result = run_mixture(
            mixture_name, families, args, dataset, state_pool, state_dim, action_dim,
            device, output_dir,
        )
        results.append(result)
        LOGGER.info("Mixture %s result: %s", mixture_name, result)

    report_path = output_dir / "prior_mixture_ablation.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "domain": args.domain,
                "seed": args.seed,
                "vae_steps": args.vae_steps,
                "rl_steps": args.rl_steps,
                "results": results,
            },
            f,
            indent=2,
        )
    LOGGER.info("Saved ablation report to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
