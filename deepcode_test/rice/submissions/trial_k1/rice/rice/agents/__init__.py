"""RICE agent subpackage.

This module exposes the core agent components used by the RICE framework:

* :class:`TargetAgent` and :class:`TargetAgentConfig` for pre-trained policy
  wrappers and SB3 training helpers.
* :class:`MaskNetwork`, :class:`PerturbedPolicy`, and :class:`MaskTrainingConfig`
  for the critical-state explanation module.
* :class:`RNDModule` and :class:`RNDRewardWrapper` for the RND exploration bonus.
* Training utilities such as :func:`train_target_agent_sb3`,
  :func:`train_mask_network`, and :func:`make_rnd_module`.
"""

from __future__ import annotations

from rice.agents.target_agent import (
    TargetAgent,
    TargetAgentConfig,
    default_cage_config,
    default_malware_config,
    default_metadrive_config,
    default_mujoco_config,
    default_selfish_mining_config,
    evaluate_target_agent,
    train_target_agent_custom,
    train_target_agent_sb3,
)

try:
    from rice.agents.mask_network import (
        MaskNetwork,
        MaskTrainingConfig,
        PerturbedPolicy,
        collect_masked_rollouts,
        default_mask_config,
        extract_critical_states,
        load_mask_network,
        make_mask_network,
        train_mask_network,
    )
except Exception:  # pragma: no cover - optional dependency guard
    MaskNetwork = None  # type: ignore
    MaskTrainingConfig = None  # type: ignore
    PerturbedPolicy = None  # type: ignore
    collect_masked_rollouts = None  # type: ignore
    default_mask_config = None  # type: ignore
    extract_critical_states = None  # type: ignore
    load_mask_network = None  # type: ignore
    make_mask_network = None  # type: ignore
    train_mask_network = None  # type: ignore

try:
    from rice.agents.rnd_network import (
        RNDModule,
        RNDRewardWrapper,
        default_rnd_config,
        make_rnd_module,
    )
except Exception:  # pragma: no cover - optional dependency guard
    RNDModule = None  # type: ignore
    RNDRewardWrapper = None  # type: ignore
    default_rnd_config = None  # type: ignore
    make_rnd_module = None  # type: ignore

__all__ = [
    # Target agent
    "TargetAgent",
    "TargetAgentConfig",
    "train_target_agent_sb3",
    "train_target_agent_custom",
    "evaluate_target_agent",
    "default_mujoco_config",
    "default_selfish_mining_config",
    "default_cage_config",
    "default_metadrive_config",
    "default_malware_config",
    # Mask network / explanation
    "MaskNetwork",
    "PerturbedPolicy",
    "MaskTrainingConfig",
    "make_mask_network",
    "train_mask_network",
    "collect_masked_rollouts",
    "extract_critical_states",
    "load_mask_network",
    "default_mask_config",
    # RND exploration bonus
    "RNDModule",
    "RNDRewardWrapper",
    "make_rnd_module",
    "default_rnd_config",
]
