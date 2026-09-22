"""Environment package for SAPG.

Provides task environments (AllegroKuka, Shadow Hand, Allegro Hand) and
wrappers used by the training loop.
"""

from .wrappers import (
    EnvWrapper,
    NormalizeObsWrapper,
    BlockAssignmentWrapper,
    CurriculumWrapper,
    make_env,
)

__all__ = [
    "EnvWrapper",
    "NormalizeObsWrapper",
    "BlockAssignmentWrapper",
    "CurriculumWrapper",
    "make_env",
    "make_allegrokuka",
    "make_shadow_hand",
    "make_allegro_hand",
]


def make_allegrokuka(task="regrasping", **kwargs):
    """Factory for AllegroKuka environments."""
    from .allegrokuka import AllegroKukaEnv

    return AllegroKukaEnv(task=task, **kwargs)


def make_shadow_hand(**kwargs):
    """Factory for Shadow Hand environments."""
    from .shadow_hand import ShadowHandEnv

    return ShadowHandEnv(**kwargs)


def make_allegro_hand(**kwargs):
    """Factory for Allegro Hand environments."""
    from .allegro_hand import AllegroHandEnv

    return AllegroHandEnv(**kwargs)
