"""Shared pytest configuration for the test suite.

This module establishes a *hermetic* baseline so the suite produces the same
result on any machine — a developer laptop, CI, or a freshly provisioned
server — regardless of what ``config/secrets.yaml`` or local services happen to
contain.

Why this exists
---------------
Without forcing configuration here, tests silently inherit production values
from ``config/secrets.yaml``, which lets "works on my machine" config drift
hide bugs until deployment, and — worse — lets a developer's real cloud and
provider credentials reach a live endpoint from a unit test run.

The V1 API this file was originally written for has been removed. What remains
is the part that is still load-bearing: the outbound-egress and model-selection
baseline, which ``Agent/DeepEvol/config/settings.py`` and the deployment
contract checks still read. The V1 pieces that went with it were a shared
SQLite test database, a ``get_settings`` cache-reset fixture, and a billing
catalog snapshot/restore fixture; all three addressed V1-only global state.
"""

from __future__ import annotations

import os

from tests.support.hermetic_environment import apply_hermetic_environment

# --- Hermetic environment baseline -----------------------------------------
# Set BEFORE any application module is imported. conftest.py is loaded by
# pytest before the test modules, so settings pick these up on first read.
# Assignment is deliberate: ``setdefault`` would preserve a developer's real
# credential and contradict the hermetic contract above.
apply_hermetic_environment(os.environ)
