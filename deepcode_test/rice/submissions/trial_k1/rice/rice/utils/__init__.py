"""Utility package for the RICE framework.

This module exposes configuration helpers, logging utilities, and replay buffers
used by training, evaluation, and experiment scripts.
"""

try:
    from rice.utils.config import (
        DomainConfig,
        MaskConfig,
        RefineConfig,
        RNDConfig,
        TargetAgentConfig,
        get_domain_config,
        load_domain_config_from_yaml,
        load_yaml_config,
        merge_with_yaml,
        save_yaml_config,
    )
except Exception as _e:  # pragma: no cover - optional dependency guard
    DomainConfig = None  # type: ignore
    MaskConfig = None  # type: ignore
    RefineConfig = None  # type: ignore
    RNDConfig = None  # type: ignore
    TargetAgentConfig = None  # type: ignore
    get_domain_config = None  # type: ignore
    load_domain_config_from_yaml = None  # type: ignore
    load_yaml_config = None  # type: ignore
    merge_with_yaml = None  # type: ignore
    save_yaml_config = None  # type: ignore

try:
    from rice.utils.logger import Logger, ConsoleLogger, make_logger, log_system_info
except Exception as _e:  # pragma: no cover
    Logger = None  # type: ignore
    ConsoleLogger = None  # type: ignore
    make_logger = None  # type: ignore
    log_system_info = None  # type: ignore

try:
    from rice.utils.replay_buffer import (
        CriticalStateReplayBuffer,
        TrajectoryBuffer,
        Transition,
        merge_critical_state_buffers,
        trajectories_to_critical_states,
    )
except Exception as _e:  # pragma: no cover
    CriticalStateReplayBuffer = None  # type: ignore
    TrajectoryBuffer = None  # type: ignore
    Transition = None  # type: ignore
    merge_critical_state_buffers = None  # type: ignore
    trajectories_to_critical_states = None  # type: ignore

__all__ = [
    "DomainConfig",
    "MaskConfig",
    "RefineConfig",
    "RNDConfig",
    "TargetAgentConfig",
    "get_domain_config",
    "load_domain_config_from_yaml",
    "load_yaml_config",
    "merge_with_yaml",
    "save_yaml_config",
    "Logger",
    "ConsoleLogger",
    "make_logger",
    "log_system_info",
    "CriticalStateReplayBuffer",
    "TrajectoryBuffer",
    "Transition",
    "merge_critical_state_buffers",
    "trajectories_to_critical_states",
]
