"""Pytest plugin for contract-driven suites that must not pass with skips."""

from __future__ import annotations

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter is not None else []
    if not skipped or exitstatus != pytest.ExitCode.OK:
        return
    reporter.write_sep(
        "=",
        f"contract-driven gate forbids skipped tests ({len(skipped)} skip reports)",
        red=True,
    )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
