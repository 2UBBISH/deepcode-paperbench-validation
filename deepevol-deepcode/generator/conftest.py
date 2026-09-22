"""Early pytest baseline shared by all repository test trees.

This file is intentionally light: it only sets process environment that must be
in place before any FastAPI app module is imported. The heavier per-test reset
fixtures still live in ``tests/conftest.py``.
"""

from __future__ import annotations

import os
import tempfile


_API_TEST_ENV_BASELINE = {
    "DEEPEVOL_API_SMS_PROVIDER": "mock",
    "DEEPEVOL_API_AGENT_BASE_URL": "http://agent.invalid:8100",
    "DEEPEVOL_API_FILE_SERVER_BASE_URL": "http://fileserver.invalid:8200",
    "DEEPEVOL_API_INTERNAL_LLM_ENABLED": "0",
    "DEEPEVOL_API_REQUIRE_REGISTRATION_ACTIVATION_CODE": "0",
    "DEEPEVOL_API_LLM_MODELS_FILE": "config/models/llm_models.yaml",
    "DEEPEVOL_API_IMAGE_MODELS_FILE": "config/models/image_models.yaml",
}


for _key, _value in _API_TEST_ENV_BASELINE.items():
    os.environ.setdefault(_key, _value)


if "DEEPEVOL_API_DATABASE_URL" not in os.environ:
    _test_db_path = os.path.join(tempfile.gettempdir(), "deepevol_api_test.db")
    for _suffix in ("", "-wal", "-shm"):
        try:
            os.remove(_test_db_path + _suffix)
        except OSError:
            pass
    os.environ["DEEPEVOL_API_DATABASE_URL"] = f"sqlite:///{_test_db_path}"
