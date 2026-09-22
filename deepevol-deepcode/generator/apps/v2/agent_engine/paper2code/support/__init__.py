"""Dependency-free helpers copied verbatim from DeepCode.

Nothing in this package is a seam: every module is real, runnable code that
the business layer needs and that did not depend on the rest of DeepCode's
``core/``. Origin and any local edits are recorded per file:

- ``verification.py``    ← core/verification.py (working tree: carries the
                            ``resolve_project_root`` fix)
- ``sandbox.py``         ← core/harness/sandbox.py
- ``windows_sandbox.py`` ← core/harness/windows_sandbox.py
- ``command_guard.py``   ← core/harness/command_guard.py
- ``hostnames.py``       ← core/network/hostnames.py
- ``platform_compat.py`` ← core/platform_compat.py

Only import paths were rewritten (``core.harness.windows_sandbox`` →
``support.windows_sandbox``, ``core.network.hostnames`` →
``support.hostnames``).
"""
