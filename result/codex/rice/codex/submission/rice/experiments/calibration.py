"""Calibrating ``alpha`` -- the bonus that controls the mask ratio.

Section 3.3 of the paper introduces ``alpha`` to "control the bonus of blinding
the target agent when training the mask network", and Appendix C.3 reports that
the fidelity is insensitive to ``alpha`` in ``{0.01, 0.001, 0.0001}``.  That
insensitivity holds *relative to the reward scale of the application*: the
paper trains with the Stable-Baselines3 pipeline, which normalises rewards, so
``alpha = 0.01`` is about one percent of a step reward.

Because the reward scales of the applications reproduce here (and of the
simplified selfish mining MDP in particular) differ from the originals, it is
useful to pick ``alpha`` such that the mask network lands in the *interior*
regime.  Both corner solutions destroy the explanation: if ``alpha`` is too
small the mask never blinds the agent and every importance score is ≈ 1; if it
is too large the mask blinds everything and every importance score is ≈ 0.

:func:`calibrate_alpha` trains short mask networks for several candidates and
returns the one whose mask rate is closest to ``target_mask_rate``.
"""

from __future__ import annotations

from typing import Dict, Sequence

from rice.envs.registry import ENV_SPECS, make_env
from rice.explanation.mask_trainer import MaskTrainingConfig, train_mask_network
from rice.experiments.common import load_agent
from rice.policies import TorchPolicy
from rice.ppo_core import PPOConfig


DEFAULT_CANDIDATES = (0.001, 0.01, 0.02, 0.05, 0.1)


def calibrate_alpha(
    env_name: str,
    agent_path: str,
    candidates: Sequence[float] = DEFAULT_CANDIDATES,
    samples: int = 5000,
    target_mask_rate: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, object]:
    """Return the candidate ``alpha`` whose mask rate is closest to the target."""
    spec = ENV_SPECS[env_name]
    policy = TorchPolicy(load_agent(env_name, agent_path, device=device))
    results = {}
    best = None
    for alpha in candidates:
        env = make_env(env_name, seed=seed)
        config = MaskTrainingConfig(
            iterations=max(2, samples // max(1, spec.horizon)),
            rollout_length=spec.horizon,
            alpha=float(alpha),
            reward_scale=spec.mask_reward_scale,
            reward_norm_clip=spec.mask_reward_clip,
            max_samples=int(samples),
            seed=seed,
            ppo=PPOConfig(learning_rate=3e-4, n_epochs=4, batch_size=64),
        )
        out = train_mask_network(env, policy, config, verbose=False)
        mask_rate = float(out.history[-1]["mask_rate"]) if out.history else float("nan")
        score = abs(mask_rate - target_mask_rate)
        results[float(alpha)] = {"mask_rate": mask_rate, "distance": score}
        if best is None or score < results[best]["distance"]:
            best = float(alpha)
        if verbose:
            print(
                "[calibrate] {} alpha={:<6} mask_rate={:.3f}".format(
                    env_name, alpha, mask_rate
                ),
                flush=True,
            )
    return {
        "env": env_name,
        "samples_per_candidate": int(samples),
        "target_mask_rate": float(target_mask_rate),
        "per_alpha": results,
        "best_alpha": best,
    }
