"""Hermetic environment policy shared by default Python test entrypoints.

The ordinary test suite must never inherit credentials or opt-in switches
that can turn a deterministic test into a provider, deployment, or live
service call.  Explicit live/local smoke scripts have their own guarded
entrypoints and do not use this policy.
"""

from __future__ import annotations

from collections.abc import MutableMapping


HERMETIC_ENV_OVERRIDES: dict[str, str] = {
    "DEEPEVOL_API_LLM_MODELS_FILE": "config/models/llm_models.yaml",
    "DEEPEVOL_API_IMAGE_MODELS_FILE": "config/models/image_models.yaml",
    "DEEPEVOL_API_LLM_MODELS_API_KEY": "test-only-llm-model-key",
    "DEEPEVOL_API_IMAGE_MODELS_API_KEY": "test-only-image-model-key",
    "OPENAI_API_KEY": " ",
    "SILICONFLOW_API_KEY": " ",
    "CUSTOM_OPENAI_API_KEY": " ",
    "DEEPEVOL_V2_SILICONFLOW_API_KEY": " ",
    "DEEPEVOL_V2_SEEDREAM_API_KEY": " ",
    "DEEPEVOL_V2_SMS_ACCESS_KEY_ID": " ",
    "DEEPEVOL_V2_SMS_ACCESS_KEY_SECRET": " ",
    # Remote Compute (experiment line) credentials: the default suite must
    # never discover, price, or rent live ECS/AutoDL resources just because a
    # developer has production keys exported.  Cloud tests monkeypatch the
    # provider client they exercise; the real-machine E2E driver sets these
    # explicitly and runs outside pytest.
    "ALIYUN_ACCESS_KEY_ID": " ",
    "ALIYUN_ACCESS_KEY_SECRET": " ",
    "ALIYUN_VSWITCH_ID": " ",
    "ALIYUN_SECURITY_GROUP_ID": " ",
    "DEEPEVOL_API_ALIYUN_ACCESS_KEY_ID": " ",
    "DEEPEVOL_API_ALIYUN_ACCESS_KEY_SECRET": " ",
    "DEEPEVOL_API_ALIYUN_VSWITCH_ID": " ",
    "DEEPEVOL_API_ALIYUN_SECURITY_GROUP_ID": " ",
    "DEEPEVOL_V2_REMOTE_COMPUTE_ALIYUN_ACCESS_KEY_ID": " ",
    "DEEPEVOL_V2_REMOTE_COMPUTE_ALIYUN_ACCESS_KEY_SECRET": " ",
    "DEEPEVOL_V2_REMOTE_COMPUTE_AUTODL_API_KEY": " ",
}

# Presence alone activates some live tests, so these values are removed
# instead of replaced with a false-looking string.
HERMETIC_ENV_REMOVALS: tuple[str, ...] = (
    "DEEPEVOL_MEMOS_LIVE_URL",
    "DEEPEVOL_MEMOS_LIVE_ALLOW_REMOTE",
    "DEEPEVOL_RUN_LOCAL_ADMIN_MODEL_RECOVERY",
    "DEEPEVOL_RUN_LOCAL_CROSS_SERVICE_SMOKE",
    "DEEPEVOL_RUN_LOCAL_FAULT_TESTS",
    "DEEPEVOL_RUN_LOCAL_REAL_MODEL",
    "DEEPEVOL_RUN_LIVE_TESTS",
    "DEEPEVOL_LIVE_MODEL",
    "DEEPEVOL_LIVE_SMS",
    "DEEPEVOL_DEPLOY_SMOKE",
    "DEEPEVOL_DEPLOY_SITE_URL",
    "DEEPEVOL_RUN_STRESS_TESTS",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
)


def apply_hermetic_environment(values: MutableMapping[str, str]) -> None:
    """Force safe defaults and remove every ordinary-suite live switch."""

    values.update(HERMETIC_ENV_OVERRIDES)
    for name in HERMETIC_ENV_REMOVALS:
        values.pop(name, None)


__all__ = [
    "HERMETIC_ENV_OVERRIDES",
    "HERMETIC_ENV_REMOVALS",
    "apply_hermetic_environment",
]
