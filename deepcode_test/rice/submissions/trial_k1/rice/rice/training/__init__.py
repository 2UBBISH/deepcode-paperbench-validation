"""Training sub-package for RICE.

This module exposes the high-level training routines used to produce target
agents, train the MaskNet explanation network, and refine agents via the RICE
pipeline.
"""

try:
    from rice.training.train_target import (
        DOMAIN_TRAINERS,
        train_cage_target,
        train_malware_target,
        train_metadrive_target,
        train_mujoco_target,
        train_selfish_mining_target,
        train_target_agent,
    )
except Exception:  # pragma: no cover - optional dependency guard
    DOMAIN_TRAINERS = None  # type: ignore
    train_cage_target = None  # type: ignore
    train_malware_target = None  # type: ignore
    train_metadrive_target = None  # type: ignore
    train_mujoco_target = None  # type: ignore
    train_selfish_mining_target = None  # type: ignore
    train_target_agent = None  # type: ignore

try:
    from rice.training.train_mask import (
        DOMAIN_MASK_TRAINERS,
        train_cage_mask,
        train_malware_mask,
        train_metadrive_mask,
        train_mujoco_mask,
        train_selfish_mining_mask,
        train_mask,
    )
except Exception:  # pragma: no cover - optional dependency guard
    DOMAIN_MASK_TRAINERS = None  # type: ignore
    train_cage_mask = None  # type: ignore
    train_malware_mask = None  # type: ignore
    train_metadrive_mask = None  # type: ignore
    train_mujoco_mask = None  # type: ignore
    train_selfish_mining_mask = None  # type: ignore
    train_mask = None  # type: ignore

try:
    from rice.training.refine_agent import (
        RefineConfig,
        default_refine_config,
        extract_critical_states_for_refining,
        load_refined_agent,
        refine_agent,
        refine_agent_vec,
    )
except Exception:  # pragma: no cover - optional dependency guard
    RefineConfig = None  # type: ignore
    default_refine_config = None  # type: ignore
    extract_critical_states_for_refining = None  # type: ignore
    load_refined_agent = None  # type: ignore
    refine_agent = None  # type: ignore
    refine_agent_vec = None  # type: ignore

__all__ = [
    # train_target
    "DOMAIN_TRAINERS",
    "train_cage_target",
    "train_malware_target",
    "train_metadrive_target",
    "train_mujoco_target",
    "train_selfish_mining_target",
    "train_target_agent",
    # train_mask
    "DOMAIN_MASK_TRAINERS",
    "train_cage_mask",
    "train_malware_mask",
    "train_metadrive_mask",
    "train_mujoco_mask",
    "train_selfish_mining_mask",
    "train_mask",
    # refine_agent
    "RefineConfig",
    "default_refine_config",
    "extract_critical_states_for_refining",
    "load_refined_agent",
    "refine_agent",
    "refine_agent_vec",
]
