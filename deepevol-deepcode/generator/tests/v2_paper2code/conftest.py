"""Process-wide knobs the driver applies with ``setdefault`` (config.apply_env_defaults) would otherwise leak from
one test into the next. Paper fidelity (ADR 0003) is the one that changes behaviour: with it on, ``phase_implement``
rejects a free-text blueprint and the implementation gate demands a passed audit — the offline fakes write free-text
plans and never call ``read_paper``. Off by default here; the fidelity tests switch it on themselves."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _paper_fidelity_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPCODE_PAPER_FIDELITY", "0")
