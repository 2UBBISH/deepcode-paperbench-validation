"""Environment zoo used by the RICE experiments."""

from rice.envs.registry import (  # noqa: F401
    ENV_SPECS,
    EnvSpec,
    make_env,
    make_stateful_env,
    list_envs,
)
