"""Make the repository importable when scripts are executed in place.

Importing this module (``import fpc._bootstrap`` or via the ``scripts/``
entry points) prepends the repository root to ``sys.path`` so that ``import fpc``
works without installing the package.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
