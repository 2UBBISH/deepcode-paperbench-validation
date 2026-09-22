"""Baselines reproduced for the Table 1 comparison.

  * ``baselines.gc_iql``  -- Goal-Conditioned IQL
  * ``baselines.gc_bc``   -- Goal-Conditioned Behavioural Cloning
  * ``baselines.opal``    -- OPAL-style offline skill discovery

FB and SF are *not* re-implemented here: the addendum specifies that they are
trained and evaluated with ``facebookresearch/controllable_agent``, so the
repository instead provides ``scripts/run_fb_sf.sh`` plus documentation for
reproducing the two columns with that external codebase (unchanged).
"""

from baselines.gc_iql import GCIQLAgent, GCIQLConfig, GoalRelabeler
from baselines.gc_bc import GCBCAgent, GCBCConfig
from baselines.opal import OPALAgent, OPALConfig

__all__ = [
    "GCIQLAgent",
    "GCIQLConfig",
    "GoalRelabeler",
    "GCBCAgent",
    "GCBCConfig",
    "OPALAgent",
    "OPALConfig",
]
