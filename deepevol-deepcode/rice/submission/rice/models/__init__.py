"""Model definitions for RICE (target policy, mask network, RND nets).

The package hosts architecture specifications and the policy/value modules used
by both stages of RICE:

* Stage 1 (explanation) trains a *mask network* ``pi_tilde_theta`` that takes a
  state and outputs a binary action ``a_t^m in {0, 1}`` (Paper Sec. 3.3).
* Stage 2 (refining) trains the *target policy* ``pi_theta`` with PPO on the
  mixed initial state distribution (Paper Algorithm 2).

Architectures follow the addendum ("Architectures"): Stable-Baselines3 default
``MlpPolicy`` for dense/sparse MuJoCo, a ``[128, 128, 128, 128]`` MLP for
Selfish Mining, a ``[64, 64, 64]`` MLP for CAGE Challenge 2, and the DI-engine
default VAC structure for MetaDrive.  Per the addendum ("Focus on overall
results") the exact numeric results do not depend on the internal network
structure, so the architecture is exposed but generic.
"""

from .policies import (  # noqa: F401
    ARCHS,
    DEFAULT_HIDDEN_SIZES,
    MASK_ARCHS,
    POLICY_ARCHS,
    ActorCritic,
    MaskArch,
    MLP,
    build_mlp,
    build_policy,
    describe_arch,
    load_policy,
    mask_arch,
    normalize_env_key,
    policy_arch,
    sample_random_action,
    save_policy,
    sb3_net_arch,
    sb3_policy_kwargs,
)

__all__ = [
    "ARCHS",
    "DEFAULT_HIDDEN_SIZES",
    "MASK_ARCHS",
    "POLICY_ARCHS",
    "ActorCritic",
    "MaskArch",
    "MLP",
    "build_mlp",
    "build_policy",
    "describe_arch",
    "load_policy",
    "mask_arch",
    "normalize_env_key",
    "policy_arch",
    "sample_random_action",
    "save_policy",
    "sb3_net_arch",
    "sb3_policy_kwargs",
]
